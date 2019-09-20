#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn.functional as F
from classy_vision.criterions import ClassyCriterion, register_criterion


@register_criterion("soft_target_cross_entropy")
class SoftTargetCrossEntropyLoss(ClassyCriterion):
    def __init__(self, config):
        """Intializer for the soft target cross-entropy loss criterion.
        This allows the targets for the cross entropy loss to be multilabel

        Config params:
        'weight': weight of sample (not yet implemented),
        'ignore_index': sample should be ignored for loss (optional),
        'reduction': specifies reduction to apply to the output (optional),
        """
        super(SoftTargetCrossEntropyLoss, self).__init__(config)
        if "weight" in config:
            raise NotImplementedError('"weight" not implemented')
        self._ignore_index = config.get("ignore_index", -100)
        self._reduction = config.get("reduction", "mean")
        self._normalize_targets = config.get("normalize_targets", "count_based")
        if self._reduction != "mean":
            raise NotImplementedError(
                'reduction type "{}" not implemented'.format(self._reduction)
            )
        self.loss_function = _SoftTargetCrossEntropyLoss(
            ignore_index=self._ignore_index,
            normalize_targets=self._normalize_targets,
            reduction=self._reduction,
        )

    def forward(self, output, target):
        return self.loss_function(logits=output, targets=target)


class _SoftTargetCrossEntropyLoss(torch.nn.Module):
    """
    Helper function for above criterion.
    This is separated out so that it can be used on its own (as a Pytorch loss)
    """

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        ignore_index=-100,
        normalize_targets="count_based",
    ):
        """
        Soft targets loss
        loss = torch.sum(- targets * F.log_softmax(logits, -1), -1)
        """
        super(_SoftTargetCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        assert normalize_targets in [None, "count_based"]
        self.normalize_targets = normalize_targets
        assert reduction in ["mean"]
        self.reduction = reduction
        self.eps = np.finfo(np.float32).eps

    def forward(self, logits, targets):
        """
        for N examples and C classes
        - targets: N x C, the plural (targets) is intentional.
        - logits: N x C
                  these are raw outputs (without softmax/sigmoid, hence "logits")

        Target elements set to ignore_index contribute 0 loss.

        Samples where all entries are ignore_index do not contribute to the loss
        reduction.
        """
        assert (
            logits.shape == targets.shape
        ), "_SoftTargetCrossEntropyLoss requires logits and target to be same"
        valid_mask = targets != self.ignore_index
        valid_targets = targets.float() * valid_mask.float()
        if self.normalize_targets == "count_based":
            valid_targets /= self.eps + valid_targets.sum(dim=1, keepdim=True)
        per_sample_per_target_loss = -valid_targets * F.log_softmax(logits, -1)
        # perform reduction
        if self.reduction == "mean":
            per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
            # normalize based on the number of samples with > 0 non-ignored targets
            loss = per_sample_loss.sum() / torch.sum(
                (torch.sum(valid_mask, -1) > 0)
            ).clamp(min=1)
        return loss

    def __call__(self, logits, targets):
        return self.forward(logits, targets)
