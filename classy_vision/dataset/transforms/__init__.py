#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import torchvision.transforms as transforms
from classy_vision.generic.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent


TRANSFORM_REGISTRY = {}


def build_transform(transform_config: Dict[str, Any]) -> Callable:
    """
    Builds a transform, first searching for it in the registry and then in
    torchvision.transforms.
    """
    assert (
        "name" in transform_config
    ), f"name not provided for transform: {transform_config}"
    name = transform_config["name"]
    transform_args = transform_config.copy()
    del transform_args["name"]
    if name in TRANSFORM_REGISTRY:
        return TRANSFORM_REGISTRY[name](**transform_args)
    # the name should be available in torchvision.transforms
    assert hasattr(transforms, name), (
        f"{name} isn't a registered tranform"
        ", nor is it available in torchvision.transforms"
    )
    return getattr(transforms, name)(**transform_args)


def build_transforms(transforms_config: List[Dict[str, Any]]) -> Callable:
    """
    Builds a transform from the list of transform configurations.
    """
    transform_list = [build_transform(config) for config in transforms_config]
    return transforms.Compose(transform_list)


def register_transform(name: str):
    """
    Decorator to register a new transform.
    New transforms can be added with the
    :func:`~classy_vision.transforms.register_transform` function decorator.

    For example::
        @register_transform('???')
        class ???:
            (...)
            def __call__():

        @register_transform('???')
        def ???:
            (...)
            return transform

    .. note::

        All transforms must be callables which return a callable that applies the
        transform. So, either a callable class or a function which returns a callable
        transform.

    Args:
        name (str): the name of the transform.
    """

    def register_transform_cls(cls: Callable[..., Callable]):
        if name in TRANSFORM_REGISTRY:
            raise ValueError("Cannot register duplicate transform ({})".format(name))
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return register_transform_cls


# automatically import any Python files in the transforms/ directory
import_all_modules(FILE_ROOT, "classy_vision.dataset.transforms")
