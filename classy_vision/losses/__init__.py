#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
import torch.nn.modules.loss as torch_losses
from classy_vision.generic.registry_utils import import_all_modules

from .classy_loss import ClassyLoss


FILE_ROOT = Path(__file__).parent


LOSS_REGISTRY = {}
LOSS_CLASS_NAMES = set()


def build_loss(config):
    """Builds a ClassyLoss from a config.

    This assumes a 'name' key in the config which is used to determine what
    model class to instantiate. For instance, a config `{"name": "my_loss",
    "foo": "bar"}` will find a class that was registered as "my_loss"
    (see :func:`register_loss`) and call .from_config on it.

    In addition to losses registered with :func:`register_loss`, we also
    support instantiating losses available in the `torch.nn.modules.loss`
    module. Any keys in the config will get expanded to parameters of the loss
    constructor. For instance, the following call will instantiate a
    :class:`torch.nn.modules.CrossEntropyLoss`:

        build_loss({"name": "CrossEntropyLoss", "reduction": "sum"})
    """

    assert "name" in config, f"name not provided for loss: {config}"
    name = config["name"]
    if name in LOSS_REGISTRY:
        return LOSS_REGISTRY[name].from_config(config)

    # the name should be available in torch.nn.modules.loss
    assert hasattr(torch_losses, name), (
        f"{name} isn't a registered loss"
        ", nor is it available in torch.nn.modules.loss"
    )
    args = config.copy()
    del args["name"]
    if "weight" in args:
        # if we are passing weights, we need to change the weights from a list
        # to a tensor
        args["weight"] = torch.tensor(args["weight"], dtype=torch.float)
    return getattr(torch_losses, name)(**args)


def register_loss(name):
    """Decorator to register a new loss."""

    def register_loss_cls(cls):
        if name in LOSS_REGISTRY:
            raise ValueError("Cannot register duplicate optimizer ({})".format(name))
        if not issubclass(cls, ClassyLoss):
            raise ValueError(
                "Loss ({}: {}) must extend ClassyLoss".format(name, cls.__name__)
            )
        LOSS_REGISTRY[name] = cls
        LOSS_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_loss_cls


# automatically import any Python files in the losses/ directory
import_all_modules(FILE_ROOT, "classy_vision.losses")


from .barron_loss import BarronLoss  # isort:skip
from .label_smoothing_loss import LabelSmoothingCrossEntropyLoss  # isort:skip
from .multi_output_sum_loss import MultiOutputSumLoss  # isort:skip
from .soft_target_cross_entropy_loss import SoftTargetCrossEntropyLoss  # isort:skip
from .sum_arbitrary_loss import SumArbitraryLoss  # isort:skip


__all__ = [
    "BarronLoss",
    "ClassyLoss",
    "LabelSmoothingCrossEntropyLoss",
    "MultiOutputSumLoss",
    "SoftTargetCrossEntropyLoss",
    "SumArbitraryLoss",
    "build_loss",
    "register_loss",
]
