#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from . import ClassyParamScheduler, register_param_scheduler


@register_param_scheduler("cosine")
class CosineDecayParamScheduler(ClassyParamScheduler):
    """
    Decays the param value after every epoch based on cosine annealing. The lr
    decays after every epoch but remain constant for that epoch.
    See https://arxiv.org/pdf/1608.03983.pdf for details.

    Example:
      base_lr: 0.1
      min_lr = 0.0001
      num_epochs = 120
    """

    def __init__(self, config):
        super().__init__(config)
        assert (
            "base_lr" in config and "min_lr" in config
        ), "Cosine decay scheduler requires a base_lr and a min_lr"

        self._base_lr = config["base_lr"]
        self._min_lr = config["min_lr"]

    def __call__(self, where: float):
        return self._min_lr + 0.5 * (self._base_lr - self._min_lr) * (
            1 + math.cos(math.pi * where)
        )
