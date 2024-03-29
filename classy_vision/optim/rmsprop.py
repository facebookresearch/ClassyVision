#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.optim
from classy_vision.generic.util import is_pos_float

from . import ClassyOptimizer, register_optimizer


@register_optimizer("rmsprop")
class RMSProp(ClassyOptimizer):
    def __init__(
        self,
        lr: float = 0.1,
        momentum: float = 0,
        weight_decay: float = 0,
        alpha: float = 0.99,
        eps: float = 1e-8,
        centered: bool = False,
    ) -> None:
        super().__init__()

        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._alpha = alpha
        self._eps = eps
        self._centered = centered

    def prepare(self, param_groups):
        self.optimizer = torch.optim.RMSprop(
            param_groups,
            lr=self._lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
            alpha=self._alpha,
            eps=self._eps,
            centered=self._centered,
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
        config.setdefault("lr", 0.1)
        config.setdefault("momentum", 0.0)
        config.setdefault("weight_decay", 0.0)
        config.setdefault("alpha", 0.99)
        config.setdefault("eps", 1e-8)
        config.setdefault("centered", False)

        for key in ["momentum", "alpha"]:
            assert (
                config[key] >= 0.0 and config[key] < 1.0 and type(config[key]) == float
            ), f"Config must contain a '{key}' in [0, 1) for RMSProp optimizer"
        assert is_pos_float(
            config["eps"]
        ), f"Config must contain a positive 'eps' for RMSProp optimizer"
        assert isinstance(
            config["centered"], bool
        ), "Config must contain a boolean 'centered' param for RMSProp optimizer"

        return cls(
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            alpha=config["alpha"],
            eps=config["eps"],
            centered=config["centered"],
        )
