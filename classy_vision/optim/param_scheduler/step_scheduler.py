#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import ClassyParamScheduler, register_param_scheduler


@register_param_scheduler("step")
class StepParamScheduler(ClassyParamScheduler):
    """
    Takes a fixed schedule for a param value.  If the length of the
    fixed schedule is less than the number of epochs, then the epochs
    are divided evenly among the param schedule.

    Example:
      values: [0.1, 0.01, 0.001, 0.0001]
      num_epochs = 120

    Then the param value will be 0.1 for epochs 0-29, 0.01 for
    epochs 30-59, 0.001 for epoch 60-89, 0.0001 for epochs 90-119.
    """

    def __init__(self, config):
        super().__init__(config)
        assert (
            "values" in config
            and isinstance(config["values"], list)
            and len(config["values"]) > 0
        ), "Step scheduler requires a list of at least one param value"
        assert config["num_epochs"] > 0, "Num epochs must be greater than 0"
        num_epochs = config["num_epochs"]

        self._param_schedule = config["values"]
        self._warmup = "warmup" in config
        if self._warmup:
            assert isinstance(config["warmup"], dict), "Warmup must be a dict"
            for name in ["init_lr", "epochs"]:
                assert name in config["warmup"], "warmup requires parameter: %s" % name
            self._warmup_init_lr = config["warmup"]["init_lr"]
            self._warmup_len = config["warmup"]["epochs"] / num_epochs

    def __call__(self, where: float):
        if self._warmup and where < self._warmup_len + self.WHERE_EPSILON:
            # interpolate between init_lr and first lr value
            warmup_progress = where / self._warmup_len
            lr = self._param_schedule[0] * warmup_progress
            lr += self._warmup_init_lr * (1 - warmup_progress)
            return lr

        ind = int((where + self.WHERE_EPSILON) * len(self._param_schedule))
        return self._param_schedule[ind]
