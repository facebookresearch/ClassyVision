#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_dataset import ClassyDataset


FILE_ROOT = Path(__file__).parent

DATASET_REGISTRY = {}
DATASET_CLASS_NAMES = set()


def build_dataset(config, *args, **kwargs):
    return DATASET_REGISTRY[config["name"]].from_config(config, *args, **kwargs)


def get_available_splits(dataset_name):
    return DATASET_REGISTRY[dataset_name].get_available_splits()


def register_dataset(name):
    """Decorator to register a new dataset."""

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

from .classy_dataset import ClassyDataset  # isort:skip
from .classy_hmdb51 import HMDB51Dataset  # isort:skip
from .classy_kinetics400 import Kinetics400Dataset  # isort:skip
from .classy_synthetic_image import SyntheticImageClassificationDataset  # isort:skip
from .classy_synthetic_video import SyntheticVideoClassificationDataset  # isort:skip
from .classy_ucf101 import UCF101Dataset  # isort:skip
from .classy_video_dataset import ClassyVideoDataset  # isort:skip

__all__ = [
    "ClassyDataset",
    "ClassyVideoDataset",
    "HMDB51Dataset",
    "Kinetics400Dataset",
    "SyntheticImageClassificationDataset",
    "SyntheticVideoClassificationDataset",
    "UCF101Dataset",
]
