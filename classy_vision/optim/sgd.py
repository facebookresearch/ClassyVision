#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.optim
from classy_vision.generic.util import is_pos_float
from classy_vision.optim.param_scheduler import (
    ClassyParamScheduler,
    build_param_scheduler,
)

from . import ClassyOptimizer, register_optimizer


@register_optimizer("sgd")
class SGD(ClassyOptimizer):
    def __init__(
        self,
        lr_scheduler: ClassyParamScheduler,
        momentum: float = 0,
        weight_decay: float = 0,
        nesterov=False,
    ):
        super().__init__(lr_scheduler=lr_scheduler)

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def init_pytorch_optimizer(self, model):
        super().init_pytorch_optimizer(model)
        self.optimizer = torch.optim.SGD(
            self.param_groups_override,
            lr=self.lr,
            nesterov=self.nesterov,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SGD":
        """Instantiates a SGD from a configuration.

        Args:
            config: A configuration for a SGD.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SGD instance.
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
            lr_scheduler=lr_scheduler,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=config["nesterov"],
        )

    @property
    def parameters(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
        }
