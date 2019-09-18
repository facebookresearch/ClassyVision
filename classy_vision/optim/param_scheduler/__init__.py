#!/usr/bin/env python3

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_vision_param_scheduler import (  # noqa F401
    ClassyParamScheduler,
    UpdateInterval,
)


FILE_ROOT = Path(__file__).parent


PARAM_SCHEDULER_REGISTRY = {}


def build_param_scheduler(config):
    return PARAM_SCHEDULER_REGISTRY[config["name"]](config)


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
