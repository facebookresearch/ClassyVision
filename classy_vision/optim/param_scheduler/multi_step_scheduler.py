#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import math
from typing import Any, Dict, List, NamedTuple, Optional, Union

from classy_vision.generic.util import is_pos_int

from . import ClassyParamScheduler, register_param_scheduler


@register_param_scheduler("multistep")
class MultiStepParamScheduler(ClassyParamScheduler):
    """
    Takes a predefined schedule for a param value, and a list of epochs
    which stand for the upper boundary (excluded) of each range.

    Example:
      values: [0.1, 0.01, 0.001, 0.0001]
      milestones = [30, 60, 80]
      num_epochs = 120

    Then the param value will be 0.1 for epochs 0-29, 0.01 for
    epochs 30-59, 0.001 for epochs 60-79, 0.0001 for epochs after epoch 80.
    Note that the length of values must be equal to the length of milestones
    plus one.
    """

    class Warmup(NamedTuple):
        epochs: Union[int, float]
        init_lr: float

    def __init__(
        self,
        values,
        num_epochs: int,
        warmup: Warmup = None,
        milestones: Optional[List[int]] = None,
    ):
        super().__init__()
        self._param_schedule = values
        self._num_epochs = num_epochs
        self._milestones = milestones

        if milestones is None:
            # Default equispaced drop_epochs behavior
            self._milestones = []
            step_width = math.ceil(self._num_epochs / float(len(self._param_schedule)))
            for idx in range(len(self._param_schedule) - 1):
                self._milestones.append(step_width * (idx + 1))

        start_epoch = 0
        for milestone in self._milestones:
            # Do not exceed the total number of epochs
            assert milestone < self._num_epochs, (
                "Epoch milestone must be smaller than total number of epochs: num_epochs=%d, milestone=%d"
                % (self._num_epochs, milestone)
            )
            # Must be in ascending order
            assert start_epoch < milestone, (
                "Epoch milestone must be smaller than start epoch: start_epoch=%d, milestone=%d"
                % (start_epoch, milestone)
            )
            start_epoch = milestone

        self._warmup = warmup
        if self._warmup:
            self._warmup_init_lr = warmup.init_lr
            self._warmup_len = warmup.epochs / self._num_epochs

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        assert (
            "values" in config
            and isinstance(config["values"], list)
            and len(config["values"]) > 0
        ), "Non-Equi Step scheduler requires a list of at least one param value"
        assert is_pos_int(config["num_epochs"]), "Num epochs must be a positive integer"
        assert config["num_epochs"] >= len(
            config["values"]
        ), "Num epochs must be greater than param schedule"

        milestones = config.get("milestones", None)
        if "milestones" in config:
            assert (
                isinstance(config["milestones"], list)
                and len(config["milestones"]) == len(config["values"]) - 1
            ), (
                "Non-Equi Step scheduler requires a list of %d epochs"
                % (len(config["values"]) - 1)
            )
        warmup = None
        if "warmup" in config:
            assert isinstance(config["warmup"], dict), "Warmup must be a dict"
            for name in ["init_lr", "epochs"]:
                assert name in config["warmup"], "warmup requires parameter: %s" % name
            warmup = cls.Warmup(**config["warmup"])

        return cls(
            values=config["values"],
            num_epochs=config["num_epochs"],
            warmup=warmup,
            milestones=milestones,
        )

    def __call__(self, where: float):
        if self._warmup and where < self._warmup_len + self.WHERE_EPSILON:
            # interpolate between init_lr and first lr value
            warmup_progress = where / self._warmup_len
            lr = self._param_schedule[0] * warmup_progress
            lr += self._warmup_init_lr * (1 - warmup_progress)
            return lr

        epoch_num = int((where + self.WHERE_EPSILON) * self._num_epochs)
        return self._param_schedule[bisect.bisect_right(self._milestones, epoch_num)]
