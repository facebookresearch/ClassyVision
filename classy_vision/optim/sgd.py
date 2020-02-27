#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.optim

from . import ClassyOptimizer, register_optimizer


@register_optimizer("sgd")
class SGD(ClassyOptimizer):
    def __init__(
        self,
        larc_config: Dict[str, Any] = None,
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_larc: bool = False,
    ):
        super().__init__()

        self.parameters.lr = lr
        self.parameters.momentum = momentum
        self.parameters.weight_decay = weight_decay
        self.parameters.nesterov = nesterov
        self.parameters.use_larc = use_larc
        self.larc_config = larc_config

    def init_pytorch_optimizer(self, model, **kwargs):
        super().init_pytorch_optimizer(model, **kwargs)
        self.optimizer = torch.optim.SGD(
            self.param_groups_override,
            lr=self.parameters.lr,
            nesterov=self.parameters.nesterov,
            momentum=self.parameters.momentum,
            weight_decay=self.parameters.weight_decay,
        )
        if self.parameters.use_larc:
            try:
                from apex.parallel.LARC import LARC
            except ImportError:
                raise RuntimeError("Apex needed for LARC")
            self.optimizer = LARC(optimizer=self.optimizer, **self.larc_config)

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
        config.setdefault("lr", 0.1)
        config.setdefault("momentum", 0.0)
        config.setdefault("weight_decay", 0.0)
        config.setdefault("nesterov", False)
        config.setdefault("use_larc", False)
        config.setdefault(
            "larc_config", {"clip": True, "eps": 1e-08, "trust_coefficient": 0.02}
        )

        assert (
            config["momentum"] >= 0.0
            and config["momentum"] < 1.0
            and type(config["momentum"]) == float
        ), "Config must contain a 'momentum' in [0, 1) for SGD optimizer"
        assert isinstance(
            config["nesterov"], bool
        ), "Config must contain a boolean 'nesterov' param for SGD optimizer"
        assert isinstance(
            config["use_larc"], bool
        ), "Config must contain a boolean 'use_larc' param for SGD optimizer"

        return cls(
            larc_config=config["larc_config"],
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=config["nesterov"],
            use_larc=config["use_larc"],
        )
