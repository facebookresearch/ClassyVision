#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_dataset import ClassyDataset


FILE_ROOT = Path(__file__).parent

DATASET_REGISTRY = {}
DATASET_CLASS_NAMES = set()


def build_dataset(config, *args, **kwargs):
    """Builds a :class:`ClassyDataset` from a config.

    This assumes a 'name' key in the config which is used to determine what
    dataset class to instantiate. For instance, a config `{"name": "my_dataset",
    "folder": "/data"}` will find a class that was registered as "my_dataset"
    (see :func:`register_dataset`) and call .from_config on it."""
    dataset = DATASET_REGISTRY[config["name"]].from_config(config, *args, **kwargs)
    num_workers = config.get("num_workers")
    if num_workers is not None:
        dataset.set_num_workers(num_workers)
    return dataset


def register_dataset(name):
    """Registers a :class:`ClassyDataset` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    ClassyDataset from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyDataset subclass like this:

    .. code-block:: python

      @register_dataset("my_dataset")
      class MyDataset(ClassyDataset):
          ...

    To instantiate a dataset from a configuration file, see
    :func:`build_dataset`."""

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        if not issubclass(cls, ClassyDataset):
            raise ValueError(
                "Dataset ({}: {}) must extend ClassyDataset".format(name, cls.__name__)
            )
        if cls.__name__ in DATASET_CLASS_NAMES:
            raise ValueError(
                "Cannot register dataset with duplicate class name({})".format(
                    cls.__name__
                )
            )
        DATASET_REGISTRY[name] = cls
        DATASET_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_dataset_cls


# automatically import any Python files in the dataset/ directory
import_all_modules(FILE_ROOT, "classy_vision.dataset")

from .classy_cifar import CIFARDataset  # isort:skip
from .classy_hmdb51 import HMDB51Dataset  # isort:skip
from .classy_kinetics400 import Kinetics400Dataset  # isort:skip
from .classy_synthetic_image import SyntheticImageDataset  # isort:skip
from .classy_synthetic_image_streaming import (  # isort:skip
    SyntheticImageStreamingDataset,  # isort:skip
)  # isort:skip
from .classy_synthetic_video import SyntheticVideoDataset  # isort:skip
from .classy_ucf101 import UCF101Dataset  # isort:skip
from .classy_video_dataset import ClassyVideoDataset  # isort:skip
from .dataloader_async_gpu_wrapper import DataloaderAsyncGPUWrapper  # isort:skip
from .dataloader_limit_wrapper import DataloaderLimitWrapper  # isort:skip
from .dataloader_skip_none_wrapper import DataloaderSkipNoneWrapper  # isort:skip
from .dataloader_wrapper import DataloaderWrapper  # isort:skip
from .image_path_dataset import ImagePathDataset  # isort:skip

__all__ = [
    "CIFARDataset",
    "ClassyDataset",
    "ClassyVideoDataset",
    "DataloaderLimitWrapper",
    "DataloaderSkipNoneWrapper",
    "DataloaderWrapper",
    "DataloaderAsyncGPUWrapper",
    "HMDB51Dataset",
    "ImagePathDataset",
    "Kinetics400Dataset",
    "SyntheticImageDataset",
    "SyntheticImageStreamingDataset",
    "SyntheticVideoDataset",
    "UCF101Dataset",
    "build_dataset",
    "register_dataset",
]
