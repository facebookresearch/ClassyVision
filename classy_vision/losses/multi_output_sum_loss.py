#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch

from . import ClassyLoss, build_loss, register_loss


@register_loss("multi_output_sum_loss")
class MultiOutputSumLoss(ClassyLoss):
    """
    Applies the provided loss to the list of outputs (or single output) and sums
    up the losses.
    """

    def __init__(self, loss) -> None:
        super().__init__()

        self._loss = loss

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiOutputSumLoss":
        """Instantiates a MultiOutputSumLoss from a configuration.

        Args:
            config: A configuration for a MultiOutpuSumLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A MultiOutputSumLoss instance.
        """
        assert (
            type(config["loss"]) == dict
        ), "loss must be a dict containing a configuration for a registered loss"
        return cls(loss=build_loss(config["loss"]))

    def forward(self, output, target):
        if torch.is_tensor(output):
            output = [output]

        loss = 0
        for pred in output:
            loss += self._loss(pred, target)

        return loss
