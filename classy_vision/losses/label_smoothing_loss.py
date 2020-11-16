#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import numpy as np
from classy_vision.generic.util import convert_to_one_hot
from classy_vision.losses import ClassyLoss, register_loss
from classy_vision.losses.soft_target_cross_entropy_loss import (
    SoftTargetCrossEntropyLoss,
)


@register_loss("label_smoothing_cross_entropy")
class LabelSmoothingCrossEntropyLoss(ClassyLoss):
    def __init__(self, ignore_index=-100, reduction="mean", smoothing_param=None):
        """Intializer for the label smoothed cross entropy loss.
        This decreases gap between output scores and encourages generalization.
        Targets provided to forward can be one-hot vectors (NxC) or class indices (Nx1).

        This normalizes the targets to a sum of 1 based on the total count of positive
        targets for a given sample before applying label smoothing.

        Args:
            ignore_index: sample should be ignored for loss if the class is this value
            reduction: specifies reduction to apply to the output
            smoothing_param: value to be added to each target entry
        """
        super().__init__()
        self._ignore_index = ignore_index
        self._reduction = reduction
        self._smoothing_param = smoothing_param
        self.loss_function = SoftTargetCrossEntropyLoss(
            self._ignore_index, self._reduction, normalize_targets=False
        )
        self._eps = np.finfo(np.float32).eps

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LabelSmoothingCrossEntropyLoss":
        """Instantiates a LabelSmoothingCrossEntropyLoss from a configuration.

        Args:
            config: A configuration for a LabelSmoothingCrossEntropyLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A LabelSmoothingCrossEntropyLoss instance.
        """

        assert (
            "smoothing_param" in config
        ), "Label Smoothing needs a smoothing parameter"
        return cls(
            ignore_index=config.get("ignore_index", -100),
            reduction=config.get("reduction", "mean"),
            smoothing_param=config.get("smoothing_param"),
        )

    def compute_valid_targets(self, target, classes):

        """
        This function takes one-hot or index target vectors and computes valid one-hot
        target vectors, based on ignore index value
        """
        target_shape_list = list(target.size())

        valid_mask = target != self._ignore_index
        valid_targets = target.float() * valid_mask.float()

        # check if targets are inputted as class integers
        if len(target_shape_list) == 1 or (
            len(target_shape_list) == 2 and target_shape_list[1] == 1
        ):

            valid_targets = convert_to_one_hot(valid_targets.view(-1, 1), classes)
            valid_targets = valid_targets.float() * valid_mask.view(-1, 1).float()

        return valid_targets

    def smooth_targets(self, valid_targets, classes):
        """
        This function takes valid (No ignore values present) one-hot target vectors
        and computes smoothed target vectors (normalized) according to the loss's
        smoothing parameter
        """

        valid_targets /= self._eps + valid_targets.sum(dim=1, keepdim=True)
        if classes > 0:
            smoothed_targets = valid_targets + (self._smoothing_param / classes)
        smoothed_targets /= self._eps + smoothed_targets.sum(dim=1, keepdim=True)

        return smoothed_targets

    def forward(self, output, target):
        valid_targets = self.compute_valid_targets(
            target=target, classes=output.shape[1]
        )
        assert (
            valid_targets.shape == output.shape
        ), "LabelSmoothingCrossEntropyLoss requires output and target to be same size"
        smoothed_targets = self.smooth_targets(
            valid_targets=valid_targets, classes=output.shape[1]
        )
        return self.loss_function(output, smoothed_targets)
