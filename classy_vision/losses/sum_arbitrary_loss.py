#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from . import ClassyLoss, build_loss, register_loss


@register_loss("sum_arbitrary")
class SumArbitraryLoss(ClassyLoss):
    """
    Sums a collection of (weighted) torch.nn losses.

    NOTE: this applies all the losses to the same output and does not support
    taking a list of outputs as input.
    """

    def __init__(
        self, losses: List[ClassyLoss], weights: Optional[Tensor] = None
    ) -> None:
        super().__init__()
        if weights is None:
            weights = torch.ones((len(losses)))

        self.losses = losses
        self.weights = weights

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SumArbitraryLoss":
        """Instantiates a SumArbitraryLoss from a configuration.

        Args:
            config: A configuration for a SumArbitraryLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SumArbitraryLoss instance.
        """
        assert (
            type(config["losses"]) == list and len(config["losses"]) > 0
        ), "losses must be a list of registered losses with length > 0"
        assert type(config["weights"]) == list and len(config["weights"]) == len(
            config["losses"]
        ), "weights must be None or a list and have same length as losses"

        loss_modules = []
        for loss_config in config["losses"]:
            loss_modules.append(build_loss(loss_config))

        assert all(
            isinstance(loss_module, ClassyLoss) for loss_module in loss_modules
        ), "All losses must be registered, valid ClassyLosses"

        return cls(losses=loss_modules, weights=config.get("weights", None))

    def forward(self, prediction, target):
        for idx, loss in enumerate(self.losses):
            current_loss = loss(prediction, target)
            if idx == 0:
                total_loss = current_loss
            else:
                total_loss = total_loss.add(self.weights[idx], current_loss)
        return total_loss
