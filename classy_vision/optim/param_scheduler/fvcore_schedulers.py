#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import inspect
from typing import Dict, Any

from fvcore.common import param_scheduler

from . import register_param_scheduler, ClassyParamScheduler, UpdateInterval


"""
The implementation of scheduler classes are moved to fvcore.
This file creates wrappers of the fvcore implementation by adding back
classyvision functionalities.
"""


def _create_classy_scheduler_class(base_class, register_name, default_update_interval):
    """
    Add back the following functionalities to the fvcore schedulers:
    1. Add `from_config` classmethod that constructs the scheduler from a dict
    2. Add `update_interval` attribute
    3. Add the class to the scheduler registry
    """

    def from_config(cls, config: Dict[str, Any]) -> param_scheduler.ParamScheduler:
        config = copy.copy(config)
        assert register_name == config.pop("name")

        update_interval = UpdateInterval.from_config(config, default_update_interval)
        param_names = inspect.signature(base_class).parameters.keys()
        # config might contain values that are not used by constructor
        kwargs = {p: config[p] for p in param_names if p in config}

        # This argument was renamed when moving to fvcore
        if "num_updates" in param_names and "num_epochs" in config:
            kwargs["num_updates"] = config["num_epochs"]
        scheduler = cls(**kwargs)
        scheduler.update_interval = update_interval
        return scheduler

    cls = type(
        base_class.__name__,
        (base_class, ClassyParamScheduler),
        {
            "from_config": classmethod(from_config),
            "update_interval": default_update_interval,
        },
    )
    if hasattr(base_class, "__doc__"):
        cls.__doc__ = base_class.__doc__.replace("num_updates", "num_epochs")
    register_param_scheduler(register_name)(cls)
    return cls


ConstantParamScheduler = _create_classy_scheduler_class(
    param_scheduler.ConstantParamScheduler,
    "constant",
    default_update_interval=UpdateInterval.EPOCH,
)

CosineParamScheduler = _create_classy_scheduler_class(
    param_scheduler.CosineParamScheduler,
    "cosine",
    default_update_interval=UpdateInterval.STEP,
)

LinearParamScheduler = _create_classy_scheduler_class(
    param_scheduler.LinearParamScheduler,
    "linear",
    default_update_interval=UpdateInterval.STEP,
)

MultiStepParamScheduler = _create_classy_scheduler_class(
    param_scheduler.MultiStepParamScheduler,
    "multistep",
    default_update_interval=UpdateInterval.EPOCH,
)

PolynomialDecayParamScheduler = _create_classy_scheduler_class(
    param_scheduler.PolynomialDecayParamScheduler,
    "polynomial",
    default_update_interval=UpdateInterval.STEP,
)

StepParamScheduler = _create_classy_scheduler_class(
    param_scheduler.StepParamScheduler,
    "step",
    default_update_interval=UpdateInterval.EPOCH,
)

StepWithFixedGammaParamScheduler = _create_classy_scheduler_class(
    param_scheduler.StepWithFixedGammaParamScheduler,
    "step_with_fixed_gamma",
    default_update_interval=UpdateInterval.STEP,
)
