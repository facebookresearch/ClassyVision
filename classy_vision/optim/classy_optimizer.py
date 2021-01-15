#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from classy_vision.generic.util import log_class_usage

from .param_scheduler import (
    ClassyParamScheduler,
    ConstantParamScheduler,
    UpdateInterval,
)


class OptionsView:
    """Convenience object to retrieve options from the optimizer param_groups.

    For instance, to get the current learning rate in the optimizer, instead of
    traversing optimizer.param_groups and finding all values for the "lr" key,
    you can just read options_view.lr. This means we don't need to keep an
    extra copy of optimizer options (such as lr, momentum) that might become
    inconsistent with the actual values used.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __getattr__(self, name):
        values = []
        for pg in self.optimizer.param_groups:
            if name in pg:
                values.append(pg[name])

        values = set(values)
        if len(values) == 0:
            raise AttributeError
        elif len(values) == 1:
            return values.pop()

        return values


class ClassyOptimizer(ABC):
    """
    Base class for optimizers.

    This wraps a :class:`torch.optim.Optimizer` instance and provides support
    for parameter scheduling. Typical PyTorch optimizers are used like this:

        optim = SGD(model.parameters(), lr=0.1)

    but the user is responsible for updating lr over the course of training.
    ClassyOptimizers extend PyTorch optimizers and allow specifying
    ClassyParamSchedulers instead:

        optim = SGD()
        optim.set_param_groups(model.parameters(), lr=LinearParamScheduler(1, 2))

    This means that as you step through the optimizer, the learning rate will
    automatically get updated with the given schedule. To access the current
    learning rate value (or any other optimizer option), you can read
    `optim.options_view.lr`. Similar to other Classy abstractions, you can also
    instantiate ClassyOptimizers from a configuration file.
    """

    def __init__(self) -> None:
        """Constructor for ClassyOptimizer.

        :var options_view: provides convenient access to current values of
            learning rate, momentum etc.
        :var _param_group_schedulers: list of dictionaries in the param_groups
            format, containing all ClassyParamScheduler instances needed. Constant
            values are converted to ConstantParamScheduler before being inserted
            here.
        """
        self.options_view = OptionsView(self)
        self.optimizer = None
        self._param_group_schedulers = None
        log_class_usage("Optimizer", self.__class__)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyOptimizer":
        """Instantiates a ClassyOptimizer from a configuration.

        Args:
            config: A configuration for the ClassyOptimizer.

        Returns:
            A ClassyOptimizer instance.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare(self, param_groups):
        """
        Prepares the optimizer for training.

        Deriving classes should initialize the underlying PyTorch
        :class:`torch.optim.Optimizer` in this call. The param_groups argument
        follows the same format supported by PyTorch (list of parameters, or
        list of param group dictionaries).

        Warning:
            This should called only after the model has been moved to the correct
            device.
        """
        raise NotImplementedError

    def set_param_groups(self, param_groups, **kwargs):
        """
        Specifies what parameters will be optimized.

        This is the public API where users of ClassyOptimizer can specify what
        parameters will get optimized. Unlike PyTorch optimizers, we don't
        require the list of param_groups in the constructor.

        Args:
            param_groups: this is either a list of Tensors (e.g.
                model.parameters()) or a list of dictionaries. If a dictionary,
                must contain a key "params" having the same format and semantics as
                PyTorch.
        """

        def cast_param_groups(params):
            """Converts a list/dict to the PyTorch param_groups format."""

            if params is None:
                return []

            if isinstance(params, dict):
                assert "params" in params
                return [params]

            pg = list(params)
            if len(pg) == 0:
                raise ValueError("optimizer got an empty parameter list")
            if not isinstance(pg[0], dict):
                pg = [{"params": pg}]
            return pg

        self._param_group_schedulers = cast_param_groups(param_groups)

        # Convert constant values to constant param schedulers. Use kwargs
        # values as defaults.
        for pg in self._param_group_schedulers:
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    pg[k] = ConstantParamScheduler(v)
                else:
                    # This is necessary to copy values from kwargs to pg
                    pg[k] = v

            for k, v in pg.items():
                if isinstance(v, (int, float)):
                    pg[k] = ConstantParamScheduler(v)

        self.prepare(self._run_schedulers(0, None))

    def _run_schedulers(self, where: float, update_interval: Optional[UpdateInterval]):
        """Goes over schedulers and gets actual values for a particular choice of where.

        If UpdateInterval is None, updates all schedulers, regardless whether
        they are configured as epoch or step. Returns a list of dictionaries in
        the param_groups format."""

        param_groups = []
        for pg in self._param_group_schedulers:
            param_group = {}
            for k, v in pg.items():
                if k == "params":
                    param_group[k] = v
                elif update_interval is None or v.update_interval == update_interval:
                    assert isinstance(
                        v, ClassyParamScheduler
                    ), f"Elements in param_groups must inherit from ClassyParamScheduler, found: {v}"
                    param_group[k] = v(where)
            param_groups.append(param_group)

        return param_groups

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def get_classy_state(self) -> Dict[str, Any]:
        """Get the state of the ClassyOptimizer.

        The returned state is used for checkpointing.

        Returns:
            A state dictionary containing the state of the optimizer.
        """
        # The "optim" key is redundant and only kept for checkpoint
        # backwards-compatibility.
        return {"optim": self.optimizer.state_dict()}

    def set_classy_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the ClassyOptimizer.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the optimizer from a checkpoint.
        """
        self.optimizer.load_state_dict(state["optim"])

    def on_epoch(self, where: float) -> None:
        """
        Called at the end of a phase.

        Updates the param schedule at the end of a phase, till training is in progress.
        This should be called by the task at the end of every epoch to update the
        schedule of epoch based param schedulers (See
        :class:`param_scheduler.ClassyParamScheduler` for more information).

        Args:
            where: where we are in terms of training progress (output of
                :func:`tasks.ClassyTask.where`)
        """
        if where < 1:
            # do not update the schedule on final on_epoch call when where == 1
            self._update_schedule(self._run_schedulers(where, UpdateInterval.EPOCH))

    def step(
        self, *args, closure: Optional[Callable] = None, where: float = None
    ) -> None:
        """
        Perform the optimization updates for a given training step.

        The optimization options (such as learning rate) performed during this
        step will correspond to the where value given as an argument to this
        function. The exact values used can be read via the options_view
        property in the optimizer.

        Args:
            where: where we are in terms of training progress (output of
                :method:`ClassyTask.where`). Must be a float in the [0;1)
                interval; This dictates parameter scheduling;
        """
        if where is None:
            raise RuntimeError(
                "ClassyOptimizer.step requires `where` argument to be provided"
            )

        assert where >= 0 and where < 1, f"Invalid where: {where}"

        if self._param_group_schedulers is None:
            raise RuntimeError(
                "ClassyOptimizer.set_param_groups must be called before step()"
            )

        self._update_schedule(self._run_schedulers(where, UpdateInterval.STEP))

        if closure is None:
            self.optimizer.step()
        else:
            self.optimizer.step(closure)

    def _update_schedule(self, param_groups) -> None:
        """Update optimizer based on a new set of param_groups."""
        assert len(self.optimizer.param_groups) == len(param_groups)

        for group, new_group in zip(self.optimizer.param_groups, param_groups):
            assert group["params"] == new_group["params"]
            group.update(new_group)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.

        See `torch.optim.Optimizer.zero_grad <https://pytorch.org/docs/stable/
        optim.html#torch.optim.Optimizer.zero_grad>`_ for more information.
        """
        self.optimizer.zero_grad()
