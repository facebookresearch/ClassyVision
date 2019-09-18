#!/usr/bin/env python3

import torch

from . import ClassyCriterion, register_criterion


@register_criterion("barron")
class BarronLoss(ClassyCriterion):
    """
    This implements the Barron loss: https://arxiv.org/pdf/1701.03077.pdf
    """

    def __init__(self, config):
        # Infinity is a valid alpha value but is frequently a string
        config["alpha"] = float(config["alpha"])
        super(BarronLoss, self).__init__(config)
        # assertions:
        assert type(config["size_average"]) == bool
        assert type(config["alpha"]) == float
        assert type(config["c"]) == float and config["c"] > 0.0

        # set fields:
        self.size_average = config["size_average"]
        self.alpha = config["alpha"]
        self.c = config["c"]
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

    def forward(self, prediction, target):
        diff = torch.add(prediction, -target)
        loss = self._forward(diff)
        loss = loss.sum(0, keepdim=True)
        if self.size_average:
            loss.div_(prediction.size(0))
        return loss
