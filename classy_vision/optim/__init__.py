#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_optimizer import ClassyOptimizer
from .param_scheduler import build_param_scheduler


FILE_ROOT = Path(__file__).parent


OPTIMIZER_REGISTRY = {}
OPTIMIZER_CLASS_NAMES = set()


def build_optimizer(config):
    """Builds a ClassyOptimizer from a config.

    This assumes a 'name' key in the config which is used to determine what
    optimizer class to instantiate. For instance, a config `{"name": "my_optimizer",
    "foo": "bar"}` will find a class that was registered as "my_optimizer"
    (see :func:`register_optimizer`) and call .from_config on it.

    Also builds the param schedulers passed in the config and associates them with the
    optimizer. The config should contain an optional "param_schedulers" key containing a
    dictionary of param scheduler configs, keyed by the parameter they control. Adds
    "num_epochs" to each of the scheduler configs and then calls
    :func:`build_param_scheduler` on each config in the dictionary.
    """
    return OPTIMIZER_REGISTRY[config["name"]].from_config(config)


def build_optimizer_schedulers(config):
    # create a deepcopy since we will be modifying the param scheduler config
    param_scheduler_config = copy.deepcopy(config.get("param_schedulers", {}))

    # build the param schedulers
    for cfg in param_scheduler_config.values():
        cfg["num_epochs"] = config["num_epochs"]

    param_schedulers = {
        param: build_param_scheduler(cfg)
        for param, cfg in param_scheduler_config.items()
    }
    return param_schedulers


def register_optimizer(name):
    """Registers a ClassyOptimizer subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    ClassyOptimizer from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyOptimizer subclass, like this:

    .. code-block:: python

        @register_optimizer('my_optimizer')
        class MyOptimizer(ClassyOptimizer):
            ...

    To instantiate an optimizer from a configuration file, see
    :func:`build_optimizer`."""

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

from .adam import Adam  # isort:skip
from .adamw import AdamW  # isort:skip
from .rmsprop import RMSProp  # isort:skip
from .rmsprop_tf import RMSPropTF  # isort:skip
from .sgd import SGD  # isort:skip

__all__ = [
    "Adam",
    "AdamW",
    "ClassyOptimizer",
    "RMSProp",
    "RMSPropTF",
    "SGD",
    "build_optimizer",
    "build_optimizer_schedulers",
    "register_optimizer",
]
