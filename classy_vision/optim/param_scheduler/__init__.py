#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_vision_param_scheduler import (  # noqa F401
    ClassyParamScheduler,
    UpdateInterval,
)


FILE_ROOT = Path(__file__).parent


PARAM_SCHEDULER_REGISTRY = {}


def build_param_scheduler(config):
    return PARAM_SCHEDULER_REGISTRY[config["name"]].from_config(config)


def register_param_scheduler(name):
    """Decorator to register a new param scheduler."""

    def register_param_scheduler_cls(cls):
        if name in PARAM_SCHEDULER_REGISTRY:
            raise ValueError(
                "Cannot register duplicate param scheduler ({})".format(name)
            )
        if not issubclass(cls, ClassyParamScheduler):
            raise ValueError(
                "Param Scheduler ({}: {}) must extend ClassyParamScheduler".format(
                    name, cls.__name__
                )
            )
        PARAM_SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_param_scheduler_cls


# automatically import any Python files in the optim/param_scheduler/ directory
import_all_modules(FILE_ROOT, "classy_vision.optim.param_scheduler")

from .constant_scheduler import ConstantParamScheduler  # isort:skip
from .cosine_scheduler import CosineParamScheduler  # isort:skip
from .multi_step_scheduler import MultiStepParamScheduler  # isort:skip
from .polynomial_decay_scheduler import PolynomialDecayParamScheduler  # isort:skip
from .step_scheduler import StepParamScheduler  # isort:skip
from .step_with_fixed_gamma_scheduler import (  # isort:skip
    StepWithFixedGammaParamScheduler,
)

__all__ = [
    "ClassyParamScheduler",
    "ConstantParamScheduler",
    "CosineParamScheduler",
    "MultiStepParamScheduler",
    "PolynomialDecayParamScheduler",
    "StepParamScheduler",
    "StepWithFixedGammaParamScheduler",
]
