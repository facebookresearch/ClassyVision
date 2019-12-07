#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch.optim
from classy_vision.generic.util import is_pos_float
from classy_vision.optim.param_scheduler import (
    ClassyParamScheduler,
    build_param_scheduler,
)

from . import ClassyOptimizer, register_optimizer


@register_optimizer("adam")
class Adam(ClassyOptimizer):
    def __init__(
        self,
        lr_scheduler: ClassyParamScheduler,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(lr_scheduler=lr_scheduler)

        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def init_pytorch_optimizer(self, model) -> None:
        super().init_pytorch_optimizer(model)
        self.optimizer = torch.optim.Adam(
            self.param_groups_override,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Adam":
        """Instantiates a Adam from a configuration.

        Args:
            config: A configuration for a Adam.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A Adam instance.
        """
        # Default params
        config.setdefault("eps", 1e-8)
        config.setdefault("amsgrad", False)

        # Check if betas is a list and convert it to a tuple
        # since a JSON config can only have lists
        if "betas" in config and type(config["betas"]) == list:
            config["betas"] = tuple(config["betas"])

        assert (
            "lr" in config
        ), "Config must contain a learning rate 'lr' section for Adam optimizer"
        assert (
            "betas" in config
            and type(config["betas"]) == tuple
            and len(config["betas"]) == 2
            and type(config["betas"][0]) == float
            and type(config["betas"][1]) == float
            and config["betas"][0] >= 0.0
            and config["betas"][0] < 1.0
            and config["betas"][1] >= 0.0
            and config["betas"][1] < 1.0
        ), "Config must contain a tuple 'betas' in [0, 1) for Adam optimizer"
        assert "weight_decay" in config and is_pos_float(
            config["weight_decay"]
        ), "Config must contain a positive 'weight_decay' for Adam optimizer"

        lr_config = config["lr"]
        if not isinstance(lr_config, dict):
            lr_config = {"name": "constant", "value": lr_config}

        lr_config["num_epochs"] = config["num_epochs"]
        lr_scheduler = build_param_scheduler(lr_config)

        return cls(
            lr_scheduler=lr_scheduler,
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
            amsgrad=config["amsgrad"],
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        }
