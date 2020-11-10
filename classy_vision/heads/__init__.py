#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_head import ClassyHead


FILE_ROOT = Path(__file__).parent


HEAD_REGISTRY = {}
HEAD_CLASS_NAMES = set()


def register_head(name):
    """Registers a ClassyHead subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    ClassyHead from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyHead subclass, like this:

    .. code-block:: python

      @register_head("my_head")
      class MyHead(ClassyHead):
          ...

    To instantiate a head from a configuration file, see
    :func:`build_head`."""

    def register_head_cls(cls):
        if name in HEAD_REGISTRY:
            raise ValueError("Cannot register duplicate head ({})".format(name))
        if not issubclass(cls, ClassyHead):
            raise ValueError(
                "Head ({}: {}) must extend ClassyHead".format(name, cls.__name__)
            )
        if cls.__name__ in HEAD_CLASS_NAMES:
            raise ValueError(
                "Cannot register head with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        HEAD_REGISTRY[name] = cls
        HEAD_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_head_cls


def build_head(config):
    """Builds a ClassyHead from a config.

    This assumes a 'name' key in the config which is used to determine what
    head class to instantiate. For instance, a config `{"name": "my_head",
    "foo": "bar"}` will find a class that was registered as "my_head"
    (see :func:`register_head`) and call .from_config on it."""

    assert "name" in config, "Expect name in config"
    assert "unique_id" in config, "Expect a global unique id in config"
    assert config["name"] in HEAD_REGISTRY, "unknown head {}".format(config["name"])
    name = config["name"]
    head_config = copy.deepcopy(config)
    del head_config["name"]
    return HEAD_REGISTRY[name].from_config(head_config)


# automatically import any Python files in the heads/ directory
import_all_modules(FILE_ROOT, "classy_vision.heads")

from .fully_connected_head import FullyConnectedHead  # isort:skip
from .fully_convolutional_linear_head import FullyConvolutionalLinearHead  # isort:skip
from .identity_head import IdentityHead  # isort:skip
from .vision_transformer_head import VisionTransformerHead  # isort:skip


__all__ = [
    "ClassyHead",
    "FullyConnectedHead",
    "FullyConvolutionalLinearHead",
    "IdentityHead",
    "VisionTransformerHead",
    "build_head",
    "register_head",
]
