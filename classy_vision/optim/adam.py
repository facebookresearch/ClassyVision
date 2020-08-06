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

        self._lr = lr
        self._betas = betas
        self._eps = eps
        self._weight_decay = weight_decay
        self._amsgrad = amsgrad

    def prepare(self, param_groups) -> None:
        self.optimizer = torch.optim.Adam(
            param_groups,
            lr=self._lr,
            betas=self._betas,
            eps=self._eps,
            weight_decay=self._weight_decay,
            amsgrad=self._amsgrad,
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
