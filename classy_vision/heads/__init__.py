#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_head import ClassyHead


FILE_ROOT = Path(__file__).parent


HEAD_REGISTRY = {}
HEAD_CLASS_NAMES = set()


def register_head(name):
    """
    New heads can be added to classy_vision with the
    :func:`~classy_vision.heads.register_head` function decorator.
    For example::

        @register_head('classification_head')
        class FullyConnectedHead(ClassyHead):
            (...)

    .. note::

        All Heads must implement the :class:`~classy_vision.heads.ClassyHead`
        interface.

    Please see the

    Args:
        name (str): the name of the head
    """

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
    assert "name" in config, "Expect name in config"
    assert "unique_id" in config, "Expect a global unique id in config"
    assert config["name"] in HEAD_REGISTRY, "unknown head"
    name = config["name"]
    head_config = config.copy()
    del head_config["name"]
    return HEAD_REGISTRY[name].from_config(head_config)


# automatically import any Python files in the heads/ directory
import_all_modules(FILE_ROOT, "classy_vision.heads")

from .fully_connected_head import FullyConnectedHead  # isort:skip
from .identity_head import IdentityHead  # isort:skip


__all__ = [
    "ClassyHead",
    "FullyConnectedHead",
    "IdentityHead",
    "build_head",
    "register_head",
]
