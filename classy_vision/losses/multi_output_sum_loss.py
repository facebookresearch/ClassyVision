#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import ClassyLoss, build_loss, register_loss


@register_loss("multi_output_sum_loss")
class MultiOutputSumLoss(ClassyLoss):
    """
    Applies the provided loss to the list of outputs (or single output) and sums
    up the losses.
    """

    @classmethod
    def from_config(cls, config):
        assert (
            type(config["loss"]) == dict
        ), "loss must be a dict containing a configuration for a registered loss"
        return cls(loss=build_loss(config["loss"]))

    def __init__(self, loss):
        super().__init__()

        self._loss = loss

    def forward(self, output, target):
        if torch.is_tensor(output):
            output = [output]

        loss = 0
        for pred in output:
            loss += self._loss(pred, target)

        return loss
