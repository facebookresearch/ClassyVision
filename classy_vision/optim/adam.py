#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch.optim

from . import ClassyOptimizer, register_optimizer


@register_optimizer("adam")
class Adam(ClassyOptimizer):
    def __init__(
        self,
        lr: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__()

        self.parameters.lr = lr
        self.parameters.betas = betas
        self.parameters.eps = eps
        self.parameters.weight_decay = weight_decay
        self.parameters.amsgrad = amsgrad

    def init_pytorch_optimizer(self, model, **kwargs) -> None:
        super().init_pytorch_optimizer(model, **kwargs)
        self.optimizer = torch.optim.Adam(
            self.param_groups_override,
            lr=self.parameters.lr,
            betas=self.parameters.betas,
            eps=self.parameters.eps,
            weight_decay=self.parameters.weight_decay,
            amsgrad=self.parameters.amsgrad,
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
        config.setdefault("lr", 0.1)
        config.setdefault("betas", [0.9, 0.999])
        config.setdefault("eps", 1e-8)
        config.setdefault("weight_decay", 0.0)
        config.setdefault("amsgrad", False)

        # Check if betas is a list and convert it to a tuple
        # since a JSON config can only have lists
        if type(config["betas"]) == list:
            config["betas"] = tuple(config["betas"])

        assert (
            type(config["betas"]) == tuple
            and len(config["betas"]) == 2
            and type(config["betas"][0]) == float
            and type(config["betas"][1]) == float
            and config["betas"][0] >= 0.0
            and config["betas"][0] < 1.0
            and config["betas"][1] >= 0.0
            and config["betas"][1] < 1.0
        ), "Config must contain a tuple 'betas' in [0, 1) for Adam optimizer"

        return cls(
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
            amsgrad=config["amsgrad"],
        )
