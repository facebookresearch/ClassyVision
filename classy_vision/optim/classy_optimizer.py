#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .param_scheduler.classy_vision_param_scheduler import UpdateInterval


class ClassyOptimizer:
    def __init__(self, lr_scheduler):
        """
        Classy Optimizer constructor.
        """
        self.lr_scheduler = lr_scheduler
        self.lr = self.lr_scheduler(0)
        self._optimizer = None
        self.optimizer_params = None

    def _validate_and_get_optimizer_params(self, model):
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
    def from_config(cls, config):
        raise NotImplementedError

    @property
    def optimizer(self):
        """
        Return a torch.optim.optimizer.Optimizer instance.
        """
        if self._optimizer is None:
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError("_optimizer must be an instance of torch.optim.Optimizer")
        return self._optimizer

    @property
    def parameters(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of parameters, e.g., with a
        different learning rate.
        """
        return {"lr": self.lr}

    def init_pytorch_optimizer(self, model):
        """
        Using the provided model, create param groups for the optimizer
        with a weight decay override for params which should be left unregularized.

        This will be called only after the model has been moved to the correct device.

        NOTE: Deriving classes should initialize the underlying Pytorch optimizer
        in this call. The simplest way to do this after a call to
        super().init_pytorch_optimizer().
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

    def get_classy_state(self):
        """
        Return the optimizer's state dict.
        """
        return {"optim": self.optimizer.state_dict(), "parameters": self.parameters}

    def set_classy_state(self, state):
        self.optimizer.load_state_dict(state["optim"])
        for param_name, param_value in state["parameters"].items():
            setattr(self, param_name, param_value)

    def backward(self, loss):
        # TODO (aadcock): Add gradient accumulation logic
        self.zero_grad()
        loss.backward()

    def update_schedule_on_epoch(self, where):
        assert self.lr_scheduler.update_interval in [
            UpdateInterval.EPOCH,
            UpdateInterval.STEP,
        ]

        if self.lr_scheduler.update_interval == UpdateInterval.EPOCH:
            self._update_schedule(where)

    def update_schedule_on_step(self, where):
        assert self.lr_scheduler.update_interval in [
            UpdateInterval.EPOCH,
            UpdateInterval.STEP,
        ]

        if self.lr_scheduler.update_interval == UpdateInterval.STEP:
            self._update_schedule(where)

    def _update_schedule(self, where):
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

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        self.optimizer.step(closure)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()
