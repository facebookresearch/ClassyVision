#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional

import torch
from classy_vision.models import ClassyModel

from .param_scheduler.classy_vision_param_scheduler import (
    ClassyParamScheduler,
    UpdateInterval,
)


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

    def __init__(self, lr_scheduler: ClassyParamScheduler):
        """
        Constructor for ClassyOptimizer.

        Args:
            lr_scheduler: The learning rate scheduler to use.
        """
        self.lr_scheduler = lr_scheduler
        self.lr = self.lr_scheduler(0)
        self.optimizer = None
        self.optimizer_params = None

    def _validate_and_get_optimizer_params(self, model: ClassyModel) -> Dict[str, Any]:
        """
        Validate and return the optimizer params.

        The optimizer params are fetched from
        :fun:`models.ClassyModel.get_optimizer_params`.

        Args:
            model: The model to get the params from.

        Returns:
            A dict containing "regularized_params" and "unregularized_params".
            Weight decay will only be applied to "regularized_params".
        """
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            optimizer_params = model.module.get_optimizer_params()
        else:
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyOptimizer":
        """Instantiates a ClassyOptimizer from a configuration.

        Args:
            config: A configuration for the ClassyOptimizer.

        Returns:
            A ClassyOptimizer instance.
        """
        raise NotImplementedError

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get the parameters of the optimizer which need to be overridden. All optimizer
        param groups will use these parameters.

        Returns:
            A kwarg dictionary that will be used to override optimizer args.
        """
        return {"lr": self.lr}

    def init_pytorch_optimizer(self, model: ClassyModel) -> None:
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
        self.optimizer_params = self._validate_and_get_optimizer_params(model)

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
        return {"optim": self.optimizer.state_dict(), "parameters": self.parameters}

    def set_classy_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the ClassyOptimizer.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the optimizer from a checkpoint.
        """
        self.optimizer.load_state_dict(state["optim"])
        for param_name, param_value in state["parameters"].items():
            setattr(self, param_name, param_value)

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
        assert self.lr_scheduler.update_interval in [
            UpdateInterval.EPOCH,
            UpdateInterval.STEP,
        ]

        if self.lr_scheduler.update_interval == UpdateInterval.EPOCH:
            self._update_schedule(where)

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
        assert self.lr_scheduler.update_interval in [
            UpdateInterval.EPOCH,
            UpdateInterval.STEP,
        ]

        if self.lr_scheduler.update_interval == UpdateInterval.STEP:
            self._update_schedule(where)

    def _update_schedule(self, where: float) -> None:
        """
        Args:
            where: where we are in terms of training progress (output of
                :func:`tasks.ClassyTask.where`)
        """
        self.lr = self.lr_scheduler(where)
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
        self.optimizer.step(closure)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.

        See `torch.optim.Optimizer.zero_grad <https://pytorch.org/docs/stable/
        optim.html#torch.optim.Optimizer.zero_grad>`_ for more information.
        """
        self.optimizer.zero_grad()
