#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules
from classy_vision.heads import build_head

from .classy_vision_model import ClassyVisionModel


FILE_ROOT = Path(__file__).parent


MODEL_REGISTRY = {}
MODEL_CLASS_NAMES = set()


def register_model(name):
    """
    New models can be added to classy_vision with the
    :func:`~classy_vision.models.register_model` function decorator.
    For example::

        @register_model('resnet')
        class ResidualNet(ClassyVisionModel):
            (...)

    .. note::

        All Models must implement the :class:`~classy_vision.models.ClassyVisionModel`
        interface.

    Please see the

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, ClassyVisionModel):
            raise ValueError(
                "Model ({}: {}) must extend ClassyVisionModel".format(
                    name, cls.__name__
                )
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


def build_model(model_config):
    assert model_config["name"] in MODEL_REGISTRY, "unknown model"
    model = MODEL_REGISTRY[model_config["name"]](model_config)
    if "heads" in model_config:
        assert (
            "freeze_trunk" in model_config
        ), "Expect freeze_trunk to be specified in config when there is head"
        heads = defaultdict(dict)
        for head_config in model_config["heads"]:
            assert "fork_block" in head_config, "Expect fork_block in config"
            fork_block = head_config["fork_block"]
            updated_config = head_config.copy()
            del updated_config["fork_block"]

            head = build_head(updated_config)
            heads[fork_block][head.unique_id] = head
        model.set_heads(heads, model_config["freeze_trunk"])
    return model


# automatically import any Python files in the models/ directory
import_all_modules(FILE_ROOT, "classy_vision.models")
