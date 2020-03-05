#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from . import ClassyParamScheduler, UpdateInterval, register_param_scheduler
from .step_scheduler import StepParamScheduler


@register_param_scheduler("step_with_fixed_gamma")
class StepWithFixedGammaParamScheduler(ClassyParamScheduler):
    """
    Decays the param value by gamma at equal number of steps so as to have the
    specified total number of decays.
    The schedule is updated after every train step by default.

    Example:

        .. code-block:: python

          base_value: 0.1
          gamma: 0.1
          num_decays: 3
          num_epochs: 120

    Then the param value will be 0.1 for epochs 0-29, 0.01 for
    epochs 30-59, 0.001 for epoch 60-89, 0.0001 for epochs 90-119.
    """

    def __init__(
        self,
        base_value: float,
        num_decays: int,
        gamma: float,
        num_epochs: int,
        update_interval: UpdateInterval = UpdateInterval.STEP,
    ):
        super().__init__(update_interval=update_interval)

        self.base_value = base_value
        self.num_decays = num_decays
        self.gamma = gamma
        self.num_epochs = num_epochs
        values = [base_value]
        for _ in range(num_decays):
            values.append(values[-1] * gamma)

        self._step_param_scheduler = StepParamScheduler(
            num_epochs=num_epochs, values=values
        )

        # make this a STEP scheduler
        self.update_interval = UpdateInterval.STEP

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StepWithFixedGammaParamScheduler":
        """Instantiates a StepWithFixedGammaParamScheduler from a configuration.

        Args:
            config: A configuration for a StepWithFixedGammaParamScheduler.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A StepWithFixedGammaParamScheduler instance.
        """
        for key in ["base_value", "gamma", "num_decays", "num_epochs"]:
            assert key in config, f"Step with fixed decay scheduler requires: {key}"
        for key in ["base_value", "gamma"]:
            assert (
                isinstance(config[key], (int, float)) and config[key] > 0
            ), f"{key} must be a positive number"
        for key in ["num_decays", "num_epochs"]:
            assert (
                isinstance(config[key], int) and config[key] > 0
            ), f"{key} must be a positive integer"

        return cls(
            base_value=config["base_value"],
            num_decays=config["num_decays"],
            gamma=config["gamma"],
            num_epochs=config["num_epochs"],
            update_interval=UpdateInterval.from_config(config, UpdateInterval.STEP),
        )

    def __call__(self, where: float) -> float:
        return self._step_param_scheduler(where)
