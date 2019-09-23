#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
import torch.nn.modules.loss as torch_losses
from classy_vision.generic.registry_utils import import_all_modules

from .classy_criterion import ClassyCriterion


FILE_ROOT = Path(__file__).parent


CRITERION_REGISTRY = {}
CRITERION_CLASS_NAMES = set()


def build_criterion(config):
    """
    Builds a criterion, first searching for it in the registry and then in
    torch.nn.modules.loss.
    """
    assert "name" in config, f"name not provided for criterion: {config}"
    name = config["name"]
    if name in CRITERION_REGISTRY:
        return CRITERION_REGISTRY[name].from_config(config)

    # the name should be available in torch.nn.modules.loss
    assert hasattr(torch_losses, name), (
        f"{name} isn't a registered criterion"
        ", nor is it available in torch.nn.modules.loss"
    )
    args = config.copy()
    del args["name"]
    if "weight" in args:
        # if we are passing weights, we need to change the weights from a list
        # to a tensor
        args["weight"] = torch.tensor(args["weight"], dtype=torch.float)
    return getattr(torch_losses, name)(**args)


def register_criterion(name):
    """Decorator to register a new criterion."""

    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError("Cannot register duplicate optimizer ({})".format(name))
        if not issubclass(cls, ClassyCriterion):
            raise ValueError(
                "Criterion ({}: {}) must extend ClassyCriterion".format(
                    name, cls.__name__
                )
            )
        CRITERION_REGISTRY[name] = cls
        CRITERION_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_criterion_cls


# automatically import any Python files in the criterion/ directory
import_all_modules(FILE_ROOT, "classy_vision.criterions")
