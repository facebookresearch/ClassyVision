#!/usr/bin/env python3

import torch

from . import ClassyCriterion, build_criterion, register_criterion


@register_criterion("multi_output_sum_loss")
class MultiOutputSumLoss(ClassyCriterion):
    """
    Applies the provided loss to the list of outputs (or single output) and sums
    up the losses.
    """

    def __init__(self, config):
        super().__init__(config)
        assert (
            type(config["loss"]) == dict
        ), "loss must be a dict containing a configuration for a registered loss"

        self._loss = build_criterion(self._config["loss"])

    def forward(self, output, target):
        if torch.is_tensor(output):
            output = [output]

        loss = 0
        for pred in output:
            loss += self._loss(pred, target)

        return loss
