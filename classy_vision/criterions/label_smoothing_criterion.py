#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from classy_vision.criterions import ClassyCriterion, register_criterion
from classy_vision.criterions.soft_target_cross_entropy_loss import (
    _SoftTargetCrossEntropyLoss,
)
from classy_vision.generic.util import convert_to_one_hot


@register_criterion("label_smoothing_cross_entropy")
class LabelSmoothingCrossEntropyLoss(ClassyCriterion):
    def __init__(self, config):
        """Intializer for the label smoothed cross entropy loss criterion.
        This decreases gap between output scores and encourages generalization.
        Targets provided to forward can be one-hot vectors (NxC) or class indices(Nx1)

        Config params:
        'weight': weight of sample (not yet implemented),
        'ignore_index': sample should be ignored for loss (optional),
        'smoothing_param': value to be added to each target entry
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__(config)
        assert "weight" not in config, '"weight" not implemented'
        self._ignore_index = config.get("ignore_index", -100)
        self._reduction = config.get("reduction", "mean")
        assert (
            "smoothing_param" in config
        ), "Label Smoothing needs a smoothing parameter"
        self._smoothing_param = config.get("smoothing_param")
        self.loss_function = _SoftTargetCrossEntropyLoss(
            ignore_index=self._ignore_index, normalize_targets=None
        )
        self._eps = np.finfo(np.float32).eps

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
        and computes smoothed target vectors (normalized) according to the criterion's
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
        return self.loss_function(logits=output, targets=smoothed_targets)
