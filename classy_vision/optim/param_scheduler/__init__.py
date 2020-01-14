#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict

from classy_vision.generic.registry_utils import import_all_modules

from .classy_vision_param_scheduler import (  # noqa F401
    ClassyParamScheduler,
    UpdateInterval,
)


FILE_ROOT = Path(__file__).parent


PARAM_SCHEDULER_REGISTRY = {}


def build_param_scheduler(config: Dict[str, Any]) -> ClassyParamScheduler:
    """Builds a :class:`ClassyParamScheduler` from a config.

    This assumes a 'name' key in the config which is used to determine what
    param scheduler class to instantiate. For instance, a config `{"name":
    "my_scheduler", "foo": "bar"}` will find a class that was registered as
    "my_scheduler" (see :func:`register_param_scheduler`) and call .from_config
    on it."""
    return PARAM_SCHEDULER_REGISTRY[config["name"]].from_config(config)


def register_param_scheduler(name):
    """Registers a :class:`ClassyParamScheduler` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    ClassyParamScheduler from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyParamScheduler subclass, like this:

    .. code-block:: python

        @register_param_scheduler('my_scheduler')
        class MyParamScheduler(ClassyParamScheduler):
            ...

    To instantiate a param scheduler from a configuration file, see
    :func:`build_param_scheduler`."""

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

from .composite_scheduler import CompositeParamScheduler, IntervalScaling  # isort:skip
from .constant_scheduler import ConstantParamScheduler  # isort:skip
from .cosine_scheduler import CosineParamScheduler  # isort:skip
from .linear_scheduler import LinearParamScheduler  # isort:skip
from .multi_step_scheduler import MultiStepParamScheduler  # isort:skip
from .polynomial_decay_scheduler import PolynomialDecayParamScheduler  # isort:skip
from .step_scheduler import StepParamScheduler  # isort:skip
from .step_with_fixed_gamma_scheduler import (  # isort:skip
    StepWithFixedGammaParamScheduler,
)

__all__ = [
    "ClassyParamScheduler",
    "CompositeParamScheduler",
    "ConstantParamScheduler",
    "CosineParamScheduler",
    "LinearParamScheduler",
    "MultiStepParamScheduler",
    "PolynomialDecayParamScheduler",
    "StepParamScheduler",
    "UpdateInterval",
    "IntervalScaling",
    "StepWithFixedGammaParamScheduler",
    "build_param_scheduler",
    "register_param_scheduler",
]
