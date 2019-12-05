#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.optim
from classy_vision.generic.util import is_pos_float
from classy_vision.optim.param_scheduler import build_param_scheduler

from . import ClassyOptimizer, register_optimizer
from .param_scheduler.classy_vision_param_scheduler import ClassyParamScheduler


@register_optimizer("rmsprop")
class RMSProp(ClassyOptimizer):
    def __init__(
        self,
        lr_scheduler: ClassyParamScheduler,
        momentum: float,
        weight_decay: float,
        alpha: float,
        eps: float = 1e-8,
        centered: bool = False,
    ) -> None:
        super().__init__(lr_scheduler=lr_scheduler)

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.eps = eps
        self.centered = centered

    def init_pytorch_optimizer(self, model):
        super().init_pytorch_optimizer(model)
        self.optimizer = torch.optim.RMSprop(
            self.param_groups_override,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            alpha=self.alpha,
            eps=self.eps,
            centered=self.centered,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RMSProp":
        """Instantiates a RMSProp from a configuration.

        Args:
            config: A configuration for a RMSProp.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A RMSProp instance.
        """
        # Default params
        config.setdefault("eps", 1e-8)
        config.setdefault("centered", False)

        assert (
            "lr" in config
        ), "Config must contain a learning rate 'lr' section for RMSProp optimizer"
        for key in ["momentum", "alpha"]:
            assert (
                key in config
                and config[key] >= 0.0
                and config[key] < 1.0
                and type(config[key]) == float
            ), f"Config must contain a '{key}' in [0, 1) for RMSProp optimizer"
        for key in ["weight_decay", "eps"]:
            assert key in config and is_pos_float(
                config[key]
            ), f"Config must contain a positive '{key}' for RMSProp optimizer"
        assert "centered" in config and isinstance(
            config["centered"], bool
        ), "Config must contain a boolean 'centered' param for RMSProp optimizer"

        lr_config = config["lr"]
        if not isinstance(lr_config, dict):
            lr_config = {"name": "constant", "value": lr_config}

        lr_config["num_epochs"] = config["num_epochs"]
        lr_scheduler = build_param_scheduler(lr_config)

        return cls(
            lr_scheduler=lr_scheduler,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            alpha=config["alpha"],
            eps=config["eps"],
            centered=config["centered"],
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "alpha": self.alpha,
            "eps": self.eps,
            "centered": self.centered,
        }
