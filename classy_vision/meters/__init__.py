#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_meter import ClassyMeter


FILE_ROOT = Path(__file__).parent


METER_REGISTRY = {}


def build_meter(config):
    return METER_REGISTRY[config["name"]].from_config(config)


def register_meter(name):
    """Decorator to register a new meter.
        New Meters can be added with the
        :func:`~classy_vision.meters.register_meter` function decorator.

        For example::
            @register_meter('accuracy')
            class AccuracyMeter(ClassyMeter):
                (...)

        .. note::

            All Meters must implement the :class:`~classy_vision.meters.ClassyMeter`
            interface.

        Args:
            name (str): the name of the meters.
    """

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
