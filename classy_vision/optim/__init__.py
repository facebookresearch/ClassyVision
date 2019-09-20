#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_optimizer import ClassyOptimizer


FILE_ROOT = Path(__file__).parent


OPTIMIZER_REGISTRY = {}
OPTIMIZER_CLASS_NAMES = set()


def build_optimizer(config, model):
    return OPTIMIZER_REGISTRY[config["name"]].from_config(config=config, model=model)


def register_optimizer(name):
    """Decorator to register a new optimizer."""

    def register_optimizer_cls(cls):
        if name in OPTIMIZER_REGISTRY:
            raise ValueError("Cannot register duplicate optimizer ({})".format(name))
        if not issubclass(cls, ClassyOptimizer):
            raise ValueError(
                "Optimizer ({}: {}) must extend ClassyVisionOptimizer".format(
                    name, cls.__name__
                )
            )
        if cls.__name__ in OPTIMIZER_CLASS_NAMES:
            raise ValueError(
                "Cannot register optimizer with duplicate class name({})".format(
                    cls.__name__
                )
            )
        OPTIMIZER_REGISTRY[name] = cls
        OPTIMIZER_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_optimizer_cls


# automatically import any Python files in the optim/ directory
import_all_modules(FILE_ROOT, "classy_vision.optim")
