#!/usr/bin/env python3

from typing import Any, Dict

from . import ClassyParamScheduler, UpdateInterval, register_param_scheduler
from .step_scheduler import StepParamScheduler


@register_param_scheduler("step_with_fixed_gamma")
class StepWithFixedGammaParamScheduler(ClassyParamScheduler):
    """
    Decays the param value by gamma at equal number of steps so as to have the
    specified total number of decays.
    Can also specify an optional warm up period where the lr is gradually
    ramped up to the initial lr.

    Example:
      base_lr: 0.1
      gamma: 0.1
      num_decays: 3
      num_epochs: 120

    Then the param value will be 0.1 for epochs 0-29, 0.01 for
    epochs 30-59, 0.001 for epoch 60-89, 0.0001 for epochs 90-119.
    """

    def __init__(self, config: Dict[str, Any]):
        for key in ["base_lr", "gamma", "num_decays", "num_epochs"]:
            assert key in config, f"Step with fixed decay scheduler requires: {key}"
        for key in ["base_lr", "gamma"]:
            assert (
                isinstance(config[key], (int, float)) and config[key] > 0
            ), f"{key} must be a positive number"
        for key in ["num_decays", "num_epochs"]:
            assert (
                isinstance(config[key], int) and config[key] > 0
            ), f"{key} must be a positive integer"

        base_lr = config["base_lr"]
        num_decays = config["num_decays"]
        gamma = config["gamma"]
        values = [base_lr]
        for _ in range(num_decays):
            values.append(values[-1] * gamma)

        step_config = {
            "name": "step",
            "values": values,
            "num_epochs": config["num_epochs"],
        }
        if "warmup" in config:
            step_config["warmup"] = config["warmup"]

        self._step_param_scheduler = StepParamScheduler(step_config)

        # make this a STEP scheduler
        self.update_interval = UpdateInterval.STEP

    def __call__(self, where: float) -> float:
        return self._step_param_scheduler(where)
