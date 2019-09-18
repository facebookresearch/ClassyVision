#!/usr/bin/env python3

from typing import Any, Dict

import torch.optim
from classy_vision.generic.util import is_pos_float
from classy_vision.models.classy_vision_model import ClassyVisionModel
from classy_vision.optim.param_scheduler import build_param_scheduler

from . import ClassyOptimizer, register_optimizer
from .param_scheduler.classy_vision_param_scheduler import ClassyParamScheduler


@register_optimizer("rmsprop")
class RMSProp(ClassyOptimizer):
    def __init__(
        self,
        model: ClassyVisionModel,
        lr_scheduler: ClassyParamScheduler,
        momentum: float,
        weight_decay: float,
        alpha: float,
        eps: float = 1e-8,
        centered: bool = False,
    ) -> None:
        super().__init__(model=model, lr_scheduler=lr_scheduler)

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.eps = eps
        self.centered = centered
        self._optimizer = torch.optim.RMSprop(
            self.param_groups_override,
            lr=self.lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=alpha,
            eps=eps,
            centered=centered,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any], model: ClassyVisionModel) -> "RMSProp":
        """
        Initializer for stochastic gradient descent optimizer. The config
        is expected to contain at least three keys:

        lr: float learning rate
        momentum: float momentum (should be [0, 1))
        weight_decay: float weight decay
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
            model=model,
            lr_scheduler=lr_scheduler,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            alpha=config["alpha"],
            eps=config["eps"],
            centered=config["centered"],
        )

    @property
    def optimizer_config(self) -> Dict[str, Any]:
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "alpha": self.alpha,
            "eps": self.eps,
            "centered": self.centered,
        }
