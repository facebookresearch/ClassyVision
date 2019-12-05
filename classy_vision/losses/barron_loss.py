#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch

from . import ClassyLoss, register_loss


@register_loss("barron")
class BarronLoss(ClassyLoss):
    """
    This implements the `Barron loss <https://arxiv.org/pdf/1701.03077.pdf>`_.
    """

    def __init__(self, alpha, size_average, c):
        super(BarronLoss, self).__init__()

        self.size_average = size_average
        self.alpha = alpha
        self.c = c
        self.z = max(1.0, 2.0 - self.alpha)

        # define all three losses:
        def _forward_zero(diff):
            out = diff.div(self.c).pow(2.0).mul(0.5).add(1.0).log()
            return out

        def _forward_inf(diff):
            out = 1.0 - diff.div(self.c).pow(2.0).mul(-0.5).exp()
            return out

        def _forward(diff):
            out = diff.div(self.c).pow(2.0).div(self.z).add(1.0).pow(self.alpha / 2.0)
            out.add_(-1.0).mul_(self.z / self.alpha)
            return out

        # set the correct loss:
        if self.alpha == 0.0:
            self._forward = _forward_zero
        elif self.alpha == -float("inf") or self.alpha == float("inf"):
            self._forward = _forward_inf
        else:
            self._forward = _forward

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BarronLoss":
        """Instantiates a BarronLoss from a configuration.

        Args:
            config: A configuration for a BarronLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A BarronLoss instance.
        """
        # Infinity is a valid alpha value but is frequently a string
        config["alpha"] = float(config["alpha"])
        # assertions:
        assert type(config["size_average"]) == bool
        assert type(config["alpha"]) == float
        assert type(config["c"]) == float and config["c"] > 0.0

        return cls(
            alpha=config["alpha"], size_average=config["size_average"], c=config["c"]
        )

    def forward(self, prediction, target):
        diff = torch.add(prediction, -target)
        loss = self._forward(diff)
        loss = loss.sum(0, keepdim=True)
        if self.size_average:
            loss.div_(prediction.size(0))
        return loss
