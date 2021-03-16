#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import traceback
from pathlib import Path
from typing import Any, Dict

from classy_vision.generic.registry_utils import import_all_modules
from fvcore.common.param_scheduler import ParamScheduler

from .classy_vision_param_scheduler import (  # noqa F401
    ClassyParamScheduler,
    UpdateInterval,
)


FILE_ROOT = Path(__file__).parent


PARAM_SCHEDULER_REGISTRY = {}
PARAM_SCHEDULER_REGISTRY_TB = {}


def build_param_scheduler(config: Dict[str, Any]) -> ParamScheduler:
    """Builds a :class:`ParamScheduler` from a config.

    This assumes a 'name' key in the config which is used to determine what
    param scheduler class to instantiate. For instance, a config `{"name":
    "my_scheduler", "foo": "bar"}` will find a class that was registered as
    "my_scheduler" (see :func:`register_param_scheduler`) and call .from_config
    on it."""
    return PARAM_SCHEDULER_REGISTRY[config["name"]].from_config(config)


def register_param_scheduler(name):
    """Registers a :class:`ParamScheduler` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    ParamScheduler from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ParamScheduler subclass that implements a `from_config` classmethod, like
    this:

    .. code-block:: python

        @register_param_scheduler('my_scheduler')
        class MyParamScheduler(ParamScheduler):
            ...

    To instantiate a param scheduler from a configuration file, see
    :func:`build_param_scheduler`."""

    def register_param_scheduler_cls(cls):
        if name in PARAM_SCHEDULER_REGISTRY:
            msg = "Cannot register duplicate param scheduler ({}). Already registered at \n{}\n"
            raise ValueError(msg.format(name, PARAM_SCHEDULER_REGISTRY_TB[name]))
        if not issubclass(cls, ParamScheduler):
            raise ValueError(
                "Param Scheduler ({}: {}) must extend ParamScheduler".format(
                    name, cls.__name__
                )
            )
        tb = "".join(traceback.format_stack())
        PARAM_SCHEDULER_REGISTRY[name] = cls
        PARAM_SCHEDULER_REGISTRY_TB[name] = tb
        return cls

    return register_param_scheduler_cls


# automatically import any Python files in the optim/param_scheduler/ directory
import_all_modules(FILE_ROOT, "classy_vision.optim.param_scheduler")

from .composite_scheduler import CompositeParamScheduler, IntervalScaling  # isort:skip
from .fvcore_schedulers import (
    ConstantParamScheduler,
    CosineParamScheduler,
    LinearParamScheduler,
    MultiStepParamScheduler,
    PolynomialDecayParamScheduler,
    StepParamScheduler,
    StepWithFixedGammaParamScheduler,
)  # isort:skip

__all__ = [
    "ParamScheduler",
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
