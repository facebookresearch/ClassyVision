#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import defaultdict
from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules
from classy_vision.heads import build_head

from .classy_model import ClassyModel


FILE_ROOT = Path(__file__).parent


MODEL_REGISTRY = {}
MODEL_CLASS_NAMES = set()


def register_model(name):
    """Registers a :class:`ClassyModel` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    :class:`ClassyModel` from a configuration file, even if the class itself is
    not part of the Classy Vision framework. To use it, apply this decorator to
    a ClassyModel subclass, like this:

    .. code-block:: python

      @register_model('resnet')
      class ResidualNet(ClassyModel):
         ...

    To instantiate a model from a configuration file, see
    :func:`build_model`."""

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, ClassyModel):
            raise ValueError(
                "Model ({}: {}) must extend ClassyModel".format(name, cls.__name__)
            )
        if cls.__name__ in MODEL_CLASS_NAMES:
            raise ValueError(
                "Cannot register model with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        MODEL_REGISTRY[name] = cls
        MODEL_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_model_cls


def build_model(config):
    """Builds a ClassyModel from a config.

    This assumes a 'name' key in the config which is used to determine what
    model class to instantiate. For instance, a config `{"name": "my_model",
    "foo": "bar"}` will find a class that was registered as "my_model"
    (see :func:`register_model`) and call .from_config on it."""

    assert config["name"] in MODEL_REGISTRY, "unknown model"
    model = MODEL_REGISTRY[config["name"]].from_config(config)
    if "heads" in config:
        heads = defaultdict(list)
        for head_config in config["heads"]:
            assert "fork_block" in head_config, "Expect fork_block in config"
            fork_block = head_config["fork_block"]
            updated_config = copy.deepcopy(head_config)
            del updated_config["fork_block"]

            head = build_head(updated_config)
            heads[fork_block].append(head)
        model.set_heads(heads)

    return model


# automatically import any Python files in the models/ directory
import_all_modules(FILE_ROOT, "classy_vision.models")

from .classy_block import ClassyBlock  # isort:skip
from .classy_model import (  # isort:skip
    ClassyModelWrapper,  # isort:skip
    ClassyModelHeadExecutorWrapper,  # isort:skip
)  # isort:skip
from .densenet import DenseNet  # isort:skip
from .efficientnet import EfficientNet  # isort:skip
from .lecun_normal_init import lecun_normal_init  # isort:skip
from .mlp import MLP  # isort:skip
from .regnet import RegNet  # isort:skip
from .resnet import ResNet  # isort:skip
from .resnext import ResNeXt  # isort:skip
from .resnext3d import ResNeXt3D  # isort:skip
from .squeeze_and_excitation_layer import SqueezeAndExcitationLayer  # isort:skip
from .vision_transformer import VisionTransformer  # isort:skip


__all__ = [
    "ClassyBlock",
    "ClassyModel",
    "ClassyModelHeadExecutorWrapper",
    "ClassyModelWrapper",
    "DenseNet",
    "EfficientNet",
    "MLP",
    "RegNet",
    "ResNet",
    "ResNeXt",
    "ResNeXt3D",
    "SqueezeAndExcitationLayer",
    "VisionTransformer",
    "build_model",
    "lecun_normal_init",
    "register_model",
]
