#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import NamedTuple, Optional  # , Union

from . import ClassyParamScheduler, register_param_scheduler


@register_param_scheduler("cosine")
class CosineParamScheduler(ClassyParamScheduler):
    """
    Changes the param value after every epoch based on a cosine schedule.
    Can be used for either cosine decay or cosine warmup schedules based on
    start and end values.
    See https://arxiv.org/pdf/1608.03983.pdf for details.

    Example:
      start_lr: 0.1
      end_lr: 0.0001
    """

    def __init__(self, start_lr: float, end_lr: float):
        super().__init__()
        self._start_lr = start_lr
        self._end_lr = end_lr

    @classmethod
    def from_config(cls, config):
        assert (
            "start_lr" in config and "end_lr" in config
        ), "Cosine scheduler requires a start_lr and a end_lr"

        return cls(start_lr=config["start_lr"], end_lr=config["end_lr"])

    def __call__(self, where: float):
        return self._end_lr + 0.5 * (self._start_lr - self._end_lr) * (
            1 + math.cos(math.pi * where)
        )
