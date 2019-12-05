#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from classy_vision.losses import ClassyLoss, register_loss


@register_loss("soft_target_cross_entropy")
class SoftTargetCrossEntropyLoss(ClassyLoss):
    def __init__(self, ignore_index, reduction, normalize_targets):
        """Intializer for the soft target cross-entropy loss loss.
        This allows the targets for the cross entropy loss to be multilabel

        Config params:
        'weight': weight of sample (not yet implemented),
        'ignore_index': sample should be ignored for loss (optional),
        'reduction': specifies reduction to apply to the output (optional),
        """
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self._ignore_index = ignore_index
        self._reduction = reduction
        assert normalize_targets in [None, "count_based"]
        self._normalize_targets = normalize_targets
        if self._reduction != "mean":
            raise NotImplementedError(
                'reduction type "{}" not implemented'.format(self._reduction)
            )
        self._eps = np.finfo(np.float32).eps

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SoftTargetCrossEntropyLoss":
        """Instantiates a SoftTargetCrossEntropyLoss from a configuration.

        Args:
            config: A configuration for a SoftTargetCrossEntropyLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SoftTargetCrossEntropyLoss instance.
        """

        if "weight" in config:
            raise NotImplementedError('"weight" not implemented')
        return cls(
            ignore_index=config.get("ignore_index", -100),
            reduction=config.get("reduction", "mean"),
            normalize_targets=config.get("normalize_targets", "count_based"),
        )

    def forward(self, output, target):
        """for N examples and C classes
        - output: N x C these are raw outputs (without softmax/sigmoid)
        - target: N x C corresponding targets

        Target elements set to ignore_index contribute 0 loss.

        Samples where all entries are ignore_index do not contribute to the loss
        reduction.
        """
        assert (
            output.shape == target.shape
        ), "SoftTargetCrossEntropyLoss requires output and target to be same"
        valid_mask = target != self._ignore_index
        valid_targets = target.float() * valid_mask.float()
        if self._normalize_targets == "count_based":
            valid_targets /= self._eps + valid_targets.sum(dim=1, keepdim=True)
        per_sample_per_target_loss = -valid_targets * F.log_softmax(output, -1)
        # perform reduction
        if self._reduction == "mean":
            per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
            # normalize based on the number of samples with > 0 non-ignored targets
            loss = per_sample_loss.sum() / torch.sum(
                (torch.sum(valid_mask, -1) > 0)
            ).clamp(min=1)
        return loss
