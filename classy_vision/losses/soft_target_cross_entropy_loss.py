#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict

import torch
import torch.nn.functional as F
from classy_vision.generic.util import convert_to_one_hot
from classy_vision.losses import ClassyLoss, register_loss


@register_loss("soft_target_cross_entropy")
class SoftTargetCrossEntropyLoss(ClassyLoss):
    def __init__(self, ignore_index=-100, reduction="mean", normalize_targets=True):
        """Intializer for the soft target cross-entropy loss loss.
        This allows the targets for the cross entropy loss to be multilabel

        Args:
            ignore_index: sample should be ignored for loss if the class is this value
            reduction: specifies reduction to apply to the output
            normalize_targets: whether the targets should be normalized to a sum of 1
                based on the total count of positive targets for a given sample
        """
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self._ignore_index = ignore_index
        self._reduction = reduction
        assert isinstance(normalize_targets, bool)
        self._normalize_targets = normalize_targets
        if self._reduction != "mean":
            raise NotImplementedError(
                'reduction type "{}" not implemented'.format(self._reduction)
            )
        self._eps = torch.finfo(torch.float32).eps

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SoftTargetCrossEntropyLoss":
        """Instantiates a SoftTargetCrossEntropyLoss from a configuration.

        Args:
            config: A configuration for a SoftTargetCrossEntropyLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SoftTargetCrossEntropyLoss instance.
        """

        return cls(
            ignore_index=config.get("ignore_index", -100),
            reduction=config.get("reduction", "mean"),
            normalize_targets=config.get("normalize_targets", True),
        )

    def forward(self, output, target):
        """for N examples and C classes
        - output: N x C these are raw outputs (without softmax/sigmoid)
        - target: N x C or N corresponding targets

        Target elements set to ignore_index contribute 0 loss.

        Samples where all entries are ignore_index do not contribute to the loss
        reduction.
        """
        # check if targets are inputted as class integers
        if target.ndim == 1:
            assert (
                output.shape[0] == target.shape[0]
            ), "SoftTargetCrossEntropyLoss requires output and target to have same batch size"
            target = convert_to_one_hot(target.view(-1, 1), output.shape[1])
        assert output.shape == target.shape, (
            "SoftTargetCrossEntropyLoss requires output and target to be same "
            f"shape: {output.shape} != {target.shape}"
        )
        valid_mask = target != self._ignore_index
        valid_targets = target.float() * valid_mask.float()
        if self._normalize_targets:
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
