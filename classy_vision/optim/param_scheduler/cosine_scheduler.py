#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
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

    class Warmup(NamedTuple):
        length: float  # normalizaed length in [0, 1)
        init_lr: float

    def __init__(self, start_lr: float, end_lr: float, warmup: Optional[Warmup] = None):
        super().__init__()
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._warmup = warmup
        if self._warmup:
            assert (
                warmup.length >= 0 and warmup.length < 1
            ), "warmup length can be in [0, 1)"
            if warmup.length <= self.WHERE_EPSILON:
                logging.warning(
                    "warmup length is too small and might cause numerical instability"
                )
            self._warmup_init_lr = warmup.init_lr

    @classmethod
    def from_config(cls, config):
        assert (
            "start_lr" in config and "end_lr" in config
        ), "Cosine scheduler requires a start_lr and a end_lr"

        warmup = None
        if "warmup" in config:
            assert isinstance(config["warmup"], dict), "Warmup must be a dict"
            for name in ["init_lr", "length"]:
                assert name in config["warmup"], "warmup requires parameter: %s" % name
            warmup = cls.Warmup(**config["warmup"])

        return cls(start_lr=config["start_lr"], end_lr=config["end_lr"], warmup=warmup)

    def __call__(self, where: float):
        warmup_length = self._warmup.length if self._warmup is not None else 0
        if (
            self._warmup is not None
            and where < self._warmup.length + self.WHERE_EPSILON
        ):
            # interpolate between init_lr and start_lr value
            warmup_progress = where / self._warmup.length
            lr = self._start_lr * warmup_progress
            lr += self._warmup_init_lr * (1 - warmup_progress)
            return lr
        return self._end_lr + 0.5 * (self._start_lr - self._end_lr) * (
            1 + math.cos(math.pi * (where - warmup_length) / (1 - warmup_length))
        )
