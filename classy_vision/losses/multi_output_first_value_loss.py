#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch

from . import ClassyLoss, build_loss, register_loss


@register_loss("multi_output_first_value_loss")
class MultiOutputFirstValueLoss(ClassyLoss):
    """
    Applies the provided loss to the first of outputs (or single output).
    """

    def __init__(self, loss, main_output_name="data") -> None:
        super().__init__()

        self._loss = loss
        self._main_output_name = main_output_name

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiOutputFirstValueLoss":
        """Instantiates a MultiOutputFirstValueLoss from a configuration.

        Args:
            config: A configuration for a MultiOutpuSumLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A MultiOutputFirstValueLoss instance.
        """
        assert (
            type(config["loss"]) == dict
        ), "loss must be a dict containing a configuration for a registered loss"
        return cls(
            loss=build_loss(config["loss"]),
            main_output_name=config.get("main_output_name", "data"),
        )

    def forward(self, output, target):
        if torch.is_tensor(output):
            output = [output]

        assert isinstance(
            output, (list, tuple, dict)
        ), "Model output must be a list of a tuple for MultiOutputFirstValueLoss"

        if isinstance(output, dict):
            assert self._main_output_name in output, (
                "%s must be among model outputs" % self._main_output_name
            )
            output = output[self._main_output_name]
        else:
            output = output[0]

        return self._loss(output, target)
