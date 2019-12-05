#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_meter import ClassyMeter


FILE_ROOT = Path(__file__).parent


METER_REGISTRY = {}


def build_meter(config):
    """Builds a :class:`ClassyMeter` from a config.

    This assumes a 'name' key in the config which is used to determine what
    meter class to instantiate. For instance, a config `{"name": "my_meter",
    "foo": "bar"}` will find a class that was registered as "my_meter" (see
    :func:`register_meter`) and call .from_config on it."""
    return METER_REGISTRY[config["name"]].from_config(config)


def build_meters(config):
    configs = [{"name": name, **args} for name, args in config.items()]
    return [build_meter(config) for config in configs]


def register_meter(name):
    """Registers a :class:`ClassyMeter` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    ClassyMeter from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyMeter subclass, like this:

    .. code-block:: python

        @register_meter('accuracy')
        class AccuracyMeter(ClassyMeter):
            ...

    To instantiate a meter from a configuration file, see
    :func:`build_meter`."""

    def register_meter_cls(cls):
        if name in METER_REGISTRY:
            raise ValueError("Cannot register duplicate meter ({})".format(name))
        if not issubclass(cls, ClassyMeter):
            raise ValueError(
                "Meter ({}: {}) must extend \
                ClassyMeter".format(
                    name, cls.__name__
                )
            )
        METER_REGISTRY[name] = cls
        return cls

    return register_meter_cls


# automatically import any Python files in the meters/ directory
import_all_modules(FILE_ROOT, "classy_vision.meters")

from .accuracy_meter import AccuracyMeter  # isort:skip
from .precision_meter import PrecisionAtKMeter  # isort:skip
from .recall_meter import RecallAtKMeter  # isort:skip
from .video_accuracy_meter import VideoAccuracyMeter  # isort:skip

__all__ = [
    "AccuracyMeter",
    "ClassyMeter",
    "PrecisionAtKMeter",
    "RecallAtKMeter",
    "VideoAccuracyMeter",
    "build_meter",
    "build_meters",
    "register_meter",
]
