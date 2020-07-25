#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn

from .param_scheduler import ClassyParamScheduler, UpdateInterval


class AttrDict(dict):
    """Dictionary class which also support attribute access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class ClassyOptimizer(ABC):
    """
    Base class for classy optimizers.

    This wraps a :class:`torch.optim.Optimizer` instance, handles learning
    rate scheduling by using a :class:`param_scheduler.ClassyParamScheduler`.
    Scheduling is also supported for any other hyperparameter (e.g. weight decay,
    momentum and others)

    Deriving classes can extend functionality be overriding the appropriate functions.
    """

    def __init__(self) -> None:
        """Constructor for ClassyOptimizer."""
        self.param_schedulers = {}
        self.parameters = AttrDict()
        self.optimizer = None

    def set_param_schedulers(
        self, param_schedulers: Dict[str, ClassyParamScheduler]
    ) -> "ClassyOptimizer":
        """Set the param schedulers for the Classy Optimizer

        Args:
            param_schedulers: A dictionary of :class:`ClassyParamScheduler`s containing
                the parameter scheduler to use for every parameter.

        Returns:
            self
        """
        self.param_schedulers = param_schedulers
        # initialize the parameters with a where of 0
        self.parameters.update(
            {param: scheduler(0) for param, scheduler in param_schedulers.items()}
        )
        return self

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

    def set_param_groups(self, param_groups, frozen_param_groups=None):
        """
        Specifies what parameters will be optimized.

        This is the public API where users of ClassyOptimizer can specify what
        parameters will get optimized. Unlike PyTorch optimizers, we don't
        require the list of param_groups in the constructor.

        param_groups have the same semantics/usage as PyTorch.
        frozen_param_groups are a list of param groups that won't be scheduled by
        ClassyOptimizer. This is useful, for instance, to disable weight decay on a
        subset of parameters while keeping LR scheduling on those same parameters.
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

        frozen_param_groups = cast_param_groups(frozen_param_groups)
        assert isinstance(frozen_param_groups, list)

        # _frozen_overrides is a copy of frozen_param_groups without the
        # "params" key.  We need an actual copy here because once param groups
        # are passed to the optimizer they get mutated.
        self._frozen_overrides = []
        for param_group in frozen_param_groups:
            self._frozen_overrides.append(
                {k: v for k, v in param_group.items() if k != "params"}
            )

        # The order between frozen_param_groups and param_groups here matters,
        # see _update_schedule implementation.
        self.prepare(frozen_param_groups + cast_param_groups(param_groups))

    def get_classy_state(self) -> Dict[str, Any]:
        """Get the state of the ClassyOptimizer.

        The returned state is used for checkpointing.

        Returns:
            A state dictionary containing the state of the optimizer.
        """
        return {
            "optim": self.optimizer.state_dict(),
            "parameters": dict(self.parameters),
        }

    def set_classy_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the ClassyOptimizer.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the optimizer from a checkpoint.
        """
        self.optimizer.load_state_dict(state["optim"])
        self.parameters.update(state["parameters"])

    def backward(self, loss: torch.Tensor) -> None:
        """
        Computer gradients with respect to the loss.

        Calls :func:`zero_grad` and then computes the gradient using
        `torch.Tensor.backward <https://pytorch.org/docs/stable/
        tensors.html#torch.Tensor.backward>`_. See :mod:`torch.autograd` for
        more information.
        """
        # TODO (aadcock): Add gradient accumulation logic
        self.zero_grad()
        loss.backward()

    def update_schedule_on_epoch(self, where: float) -> None:
        """
        Update the param schedule at the end of an epoch.

        This should be called by the task at the end of every epoch to update the
        schedule of epoch based param schedulers (See
        :class:`param_scheduler.ClassyParamScheduler` for more information).

        Args:
            where: where we are in terms of training progress (output of
                :func:`tasks.ClassyTask.where`)
        """
        for param, scheduler in self.param_schedulers.items():
            assert scheduler.update_interval in [
                UpdateInterval.EPOCH,
                UpdateInterval.STEP,
            ]

            if scheduler.update_interval == UpdateInterval.EPOCH:
                self.parameters[param] = scheduler(where)

        self._update_schedule()

    def update_schedule_on_step(self, where: float) -> None:
        """
        Update the param schedule at the end of a train step.

        This should be called by the task at the end of every train step (
        :func:`tasks.ClassyTask.train_step`) to update the schedule of step
        based param schedulers (See :class:`param_scheduler.ClassyParamScheduler`
        for more information).

        Args:
            where: where we are in terms of training progress (output of
                :method:`ClassyTask.where`)
        """
        for param, scheduler in self.param_schedulers.items():
            assert scheduler.update_interval in [
                UpdateInterval.EPOCH,
                UpdateInterval.STEP,
            ]

            if scheduler.update_interval == UpdateInterval.STEP:
                self.parameters[param] = scheduler(where)

        self._update_schedule()

    def _update_schedule(self) -> None:
        """Update the optimizer's parameters based on self.parameters."""
        for group in self.optimizer.param_groups:
            lr_scale = group.get("lr_scale", 1.0)
            parameters = copy.deepcopy(self.parameters)
            parameters["lr"] *= lr_scale
            group.update(parameters)

        # Here there's an assumption that pytorch optimizer maintain the order
        # of param_groups and that frozen_param_groups were added before the
        # others. This must be kept in sync with the prepare call in
        # set_param_groups
        for i, override in enumerate(self._frozen_overrides):
            self.optimizer.param_groups[i].update(**override)

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        See `torch.optim.Optimizer.step <https://pytorch.org/docs/stable/
        optim.html#torch.optim.Optimizer.step>`_for more information.

        Args:
            closure: A closure that re-evaluates the model and returns the loss
        """
        if closure is None:
            self.optimizer.step()
        else:
            self.optimizer.step(closure)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.

        See `torch.optim.Optimizer.zero_grad <https://pytorch.org/docs/stable/
        optim.html#torch.optim.Optimizer.zero_grad>`_ for more information.
        """
        self.optimizer.zero_grad()
