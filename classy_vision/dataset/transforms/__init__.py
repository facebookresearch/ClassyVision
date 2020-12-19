#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
from pathlib import Path
from typing import Any, Callable, Dict, List

import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from classy_vision.generic.registry_utils import import_all_modules
from classy_vision.generic.util import log_class_usage

from .classy_transform import ClassyTransform


FILE_ROOT = Path(__file__).parent


TRANSFORM_REGISTRY = {}


def build_transform(transform_config: Dict[str, Any]) -> Callable:
    """Builds a :class:`ClassyTransform` from a config.

    This assumes a 'name' key in the config which is used to determine what
    transform class to instantiate. For instance, a config `{"name":
    "my_transform", "foo": "bar"}` will find a class that was registered as
    "my_transform" (see :func:`register_transform`) and call .from_config on
    it.

    In addition to transforms registered with :func:`register_transform`, we
    also support instantiating transforms available in the
    `torchvision.transforms <https://pytorch.org/docs/stable/torchvision/
    transforms.html>`_ module. Any keys in the config will get expanded
    to parameters of the transform constructor. For instance, the following
    call will instantiate a :class:`torchvision.transforms.CenterCrop`:

    .. code-block:: python

      build_transform({"name": "CenterCrop", "size": 224})
    """
    assert (
        "name" in transform_config
    ), f"name not provided for transform: {transform_config}"
    name = transform_config["name"]
    transform_args = copy.deepcopy(transform_config)
    del transform_args["name"]
    if name in TRANSFORM_REGISTRY:
        transform = TRANSFORM_REGISTRY[name].from_config(transform_args)
    else:
        # the name should be available in torchvision.transforms
        # if users specify the torchvision transform name in snake case,
        # we need to convert it to title case.
        if not (hasattr(transforms, name) or hasattr(transforms_video, name)):
            name = name.title().replace("_", "")
        assert hasattr(transforms, name) or hasattr(transforms_video, name), (
            f"{name} isn't a registered tranform"
            ", nor is it available in torchvision.transforms"
        )
        if hasattr(transforms, name):
            transform = getattr(transforms, name)(**transform_args)
        else:
            transform = getattr(transforms_video, name)(**transform_args)
    log_class_usage("Transform", transform.__class__)
    return transform


def build_transforms(transforms_config: List[Dict[str, Any]]) -> Callable:
    """
    Builds a transform from the list of transform configurations.
    """
    transform_list = [build_transform(config) for config in transforms_config]
    return transforms.Compose(transform_list)


def register_transform(name: str):
    """Registers a :class:`ClassyTransform` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    :class:`ClassyTransform` from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyTransform subclass like this:

    .. code-block:: python

      @register_transform("my_transform")
      class MyTransform(ClassyTransform):
          ...

    To instantiate a transform from a configuration file, see
    :func:`build_transform`."""

    def register_transform_cls(cls: Callable[..., Callable]):
        if name in TRANSFORM_REGISTRY:
            raise ValueError("Cannot register duplicate transform ({})".format(name))
        if hasattr(transforms, name) or hasattr(transforms_video, name):
            raise ValueError(
                "{} has existed in torchvision.transforms, Please change the name!".format(
                    name
                )
            )
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return register_transform_cls


# automatically import any Python files in the transforms/ directory
import_all_modules(FILE_ROOT, "classy_vision.dataset.transforms")

from .lighting_transform import LightingTransform  # isort:skip
from .util import ApplyTransformToKey  # isort:skip
from .util import ImagenetAugmentTransform  # isort:skip
from .util import ImagenetAugmentTransform  # isort:skip
from .util import ImagenetNoAugmentTransform  # isort:skip
from .util import GenericImageTransform  # isort:skip
from .util import TupleToMapTransform  # isort:skip


__all__ = [
    "ClassyTransform",
    "ImagenetAugmentTransform",
    "ImagenetNoAugmentTransform",
    "GenericImageTransform",
    "ApplyTransformToKey",
    "TupleToMapTransform",
    "LightingTransform",
    "register_transform",
    "build_transform",
    "build_transforms",
]
