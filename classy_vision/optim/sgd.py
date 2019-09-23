#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim
from classy_vision.generic.util import is_pos_float
from classy_vision.optim.param_scheduler import build_param_scheduler

from . import ClassyOptimizer, register_optimizer


@register_optimizer("sgd")
class SGD(ClassyOptimizer):
    def __init__(self, model, lr_scheduler, momentum, weight_decay, nesterov=False):
        super().__init__(model=model, lr_scheduler=lr_scheduler)

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._optimizer = torch.optim.SGD(
            self.param_groups_override,
            lr=self.lr,
            nesterov=nesterov,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    @classmethod
    def from_config(cls, config, model):
        """
        Initializer for stochastic gradient descent optimizer. The config
        is expected to contain at least three keys:

        lr: float learning rate
        momentum: float momentum (should be [0, 1))
        weight_decay: float weight decay
        """
        # Default params
        config["nesterov"] = config.get("nesterov", False)

        assert (
            "lr" in config
        ), "Config must contain a learning rate 'lr' section for SGD optimizer"
        assert (
            "momentum" in config
            and config["momentum"] >= 0.0
            and config["momentum"] < 1.0
            and type(config["momentum"]) == float
        ), "Config must contain a 'momentum' in [0, 1) for SGD optimizer"
        assert "nesterov" in config and isinstance(
            config["nesterov"], bool
        ), "Config must contain a boolean 'nesterov' param for SGD optimizer"
        assert "weight_decay" in config and is_pos_float(
            config["weight_decay"]
        ), "Config must contain a positive 'weight_decay' for SGD optimizer"

        lr_config = config["lr"]
        if not isinstance(lr_config, dict):
            lr_config = {"name": "constant", "value": lr_config}

        lr_config["num_epochs"] = config["num_epochs"]
        lr_scheduler = build_param_scheduler(lr_config)

        return cls(
            model=model,
            lr_scheduler=lr_scheduler,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=config["nesterov"],
        )

    @property
    def hyperparameters(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
        }
