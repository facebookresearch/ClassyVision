#!/usr/bin/env python3

import torch
from classy_vision.generic.util import is_on_gpu

from . import ClassyCriterion, register_criterion


@register_criterion("sum_bce_with_logits")
class SumBCEWithLogitsLoss(ClassyCriterion):
    def __init__(self, config):
        """Intializer for the sum BCEWithLogitsLoss criterion. For a single
        tensor, this is equivalent to BCEWithLogitsLoss. For a
        list of tensors, this computes the sum of the BCEWithLogitsLoss
        losses for each tensor in the list against the target.

        Config params:
        'weight': weight of sample, optional,
        'ignore_index': sample should be ignored for loss, optional,
        'reduction': specifies reduction to apply to the output, optional,
        """
        super(SumBCEWithLogitsLoss, self).__init__(config)
        self._weight = config["weight"] if "weight" in config else None
        self._reduction = config["reduction"] if "reduction" in config else "mean"
        self._losses = torch.nn.modules.ModuleList([])

    def _create_loss_function(self):
        copy_to_gpu = is_on_gpu(self._losses)
        self._losses.append(
            torch.nn.modules.BCEWithLogitsLoss(
                weight=self._weight, reduction=self._reduction
            )
        )
        if copy_to_gpu:
            self._losses.cuda()
        return self

    def forward(self, output, target):
        if torch.is_tensor(output):
            output = [output]

        loss = 0
        for idx, pred in enumerate(output):
            while idx >= len(self._losses):
                self._create_loss_function()
            loss += self._losses[idx](pred, target)

        return loss
