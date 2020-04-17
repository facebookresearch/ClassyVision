#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Union

import torch
from classy_vision.losses import ClassyLoss
from classy_vision.models import ClassyModel
from torch import nn

from .param_scheduler import ClassyParamScheduler, UpdateInterval


class AttrDict(dict):
    """Dictionary class which also support attribute access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class ClassyOptimizer:
    """
    Base class for classy optimizers.

    This wraps a :class:`torch.optim.Optimizer` instance, handles learning
    rate scheduling by using a :class:`param_scheduler.ClassyParamScheduler`
    and supports specifying regularized and unregularized param groups.
    Specifying unregularized params is especially useful to avoid applying
    weight decay on batch norm. See
    :func:`classy_vision.models.ClassyModel.get_optimizer_params` for more
    information.

    Deriving classes can extend functionality be overriding the appropriate functions.
    """

    def __init__(self) -> None:
        """Constructor for ClassyOptimizer."""
        self.param_schedulers = {}
        self.parameters = AttrDict()
        self.optimizer = None
        self.optimizer_params = None

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

    @staticmethod
    def _validate_optimizer_params(model: Union[ClassyLoss, ClassyModel]):
        optimizer_params = model.get_optimizer_params()

        assert isinstance(optimizer_params, dict) and set(optimizer_params.keys()) == {
            "regularized_params",
            "unregularized_params",
        }, "get_optimizer_params() of {0} should return dict with exact two keys\
            'regularized_params', 'unregularized_params'".format(
            type(model).__name__
        )

        trainable_params = [
            params for params in model.parameters() if params.requires_grad
        ]
        assert len(trainable_params) == len(
            optimizer_params["regularized_params"]
        ) + len(optimizer_params["unregularized_params"]), (
            "get_optimizer_params() of {0} should return params that cover all"
            "trainable params of model".format(type(model).__name__)
        )

        return optimizer_params

    def _validate_and_get_optimizer_params(
        self, model: ClassyModel, loss: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Validate and return the optimizer params.

        The optimizer params are fetched from
        :fun:`models.ClassyModel.get_optimizer_params`.

        Args:
            model: The model to get the params from.
            loss: The loss. If present, and a ClassyLoss, then the loss may
                also contribute parameters.

        Returns:
            A dict containing "regularized_params" and "unregularized_params".
            Weight decay will only be applied to "regularized_params".
        """
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        optimizer_params = self._validate_optimizer_params(model)

        if loss is not None and isinstance(loss, ClassyLoss):
            loss_params = self._validate_optimizer_params(loss)
            # Merge loss and model params.
            optimizer_params = {
                key: value + loss_params[key]
                for (key, value) in optimizer_params.items()
            }

        return optimizer_params

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyOptimizer":
        """Instantiates a ClassyOptimizer from a configuration.

        Args:
            config: A configuration for the ClassyOptimizer.

        Returns:
            A ClassyOptimizer instance.
        """
        raise NotImplementedError

    def init_pytorch_optimizer(
        self, model: ClassyModel, loss: Optional[Union[ClassyLoss, Any]] = None
    ) -> None:
        """
        Initialize the underlying :class:`torch.optim.Optimizer` instance.

        Using the provided model, create param groups for the optimizer with a
        weight decay override for params which should be left unregularized.

        Note:
            Deriving classes should initialize the underlying Pytorch optimizer
            in this call. The simplest way to do this after a call to

            ``super().init_pytorch_optimizer()``

        Warning:
            This should called only after the model has been moved to the correct
            device.
        """
        self.optimizer_params = self._validate_and_get_optimizer_params(model, loss)

        param_groups_override = []
        self.contains_unregularized_params = False
        if len(self.optimizer_params["unregularized_params"]) != 0:
            param_groups_override.append(
                {
                    "params": self.optimizer_params["unregularized_params"],
                    "weight_decay": 0.0,
                }
            )
            self.contains_unregularized_params = True

        if len(self.optimizer_params["regularized_params"]) != 0:
            param_groups_override.append(
                {"params": self.optimizer_params["regularized_params"]}
            )
        self.param_groups_override = param_groups_override

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
            group.update(self.parameters)

        # Here there's an assumption that pytorch optimizer maintain the order of
        # param_groups and batch_norm param_group is 0th param_group as initially
        # set in the __init__ call.
        # It seems like pytorch optim doesn't have way to get params by 'id':
        #   See thread https://github.com/pytorch/pytorch/issues/1489
        if self.contains_unregularized_params:
            self.optimizer.param_groups[0].update(weight_decay=0.0)

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
