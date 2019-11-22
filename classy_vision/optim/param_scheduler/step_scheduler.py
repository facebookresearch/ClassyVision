#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, NamedTuple, Optional, Union

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

    def __init__(self, num_epochs: Union[int, float], values: List[float]):
        super().__init__()

        self._param_schedule = values

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        assert (
            "values" in config
            and isinstance(config["values"], list)
            and len(config["values"]) > 0
        ), "Step scheduler requires a list of at least one param value"
        assert config["num_epochs"] > 0, "Num epochs must be greater than 0"

        return cls(num_epochs=config["num_epochs"], values=config["values"])

    def __call__(self, where: float):
        ind = int((where + self.WHERE_EPSILON) * len(self._param_schedule))
        return self._param_schedule[ind]
