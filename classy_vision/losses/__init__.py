#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from pathlib import Path

import torch
import torch.nn.modules.loss as torch_losses
from classy_vision.generic.registry_utils import import_all_modules
from classy_vision.generic.util import log_class_usage

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
    support instantiating losses available in the `torch.nn.modules.loss <https:
    //pytorch.org/docs/stable/nn.html#loss-functions>`_
    module. Any keys in the config will get expanded to parameters of the loss
    constructor. For instance, the following call will instantiate a
    `torch.nn.modules.CrossEntropyLoss <https://pytorch.org/docs/stable/
    nn.html#torch.nn.CrossEntropyLoss>`_:

    .. code-block:: python

     build_loss({"name": "CrossEntropyLoss", "reduction": "sum"})
    """

    assert "name" in config, f"name not provided for loss: {config}"
    name = config["name"]
    args = copy.deepcopy(config)
    del args["name"]
    if "weight" in args and args["weight"] is not None:
        # if we are passing weights, we need to change the weights from a list
        # to a tensor
        args["weight"] = torch.tensor(args["weight"], dtype=torch.float)
    if name in LOSS_REGISTRY:
        loss = LOSS_REGISTRY[name].from_config(config)
    else:
        # the name should be available in torch.nn.modules.loss
        assert hasattr(torch_losses, name), (
            f"{name} isn't a registered loss"
            ", nor is it available in torch.nn.modules.loss"
        )
        loss = getattr(torch_losses, name)(**args)
    log_class_usage("Loss", loss.__class__)
    return loss


def register_loss(name):
    """Registers a ClassyLoss subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    ClassyLoss from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyLoss subclass, like this:

    .. code-block:: python

     @register_loss("my_loss")
     class MyLoss(ClassyLoss):
          ...

    To instantiate a loss from a configuration file, see
    :func:`build_loss`."""

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
