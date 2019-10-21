#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from classy_vision.generic.util import set_proxies, unset_proxies

from . import register_dataset
from .classy_dataset import ClassyDataset
from .core import WrapDataset
from .transforms.util import (
    ImagenetConstants,
    TupleToMapTransform,
    build_field_transform_default_imagenet,
)


# constants for the CIFAR datasets:
DATA_PATH = "/mnt/vol/gfsai-east/ai-group/users/lvdmaaten/cifar"
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _cifar_augment_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ImagenetConstants.MEAN, std=ImagenetConstants.STD
            ),
        ]
    )


def _cifar_no_augment_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ImagenetConstants.MEAN, std=ImagenetConstants.STD
            ),
        ]
    )


class CifarDataset(ClassyDataset):
    _CIFAR_TYPE = None

    def __init__(self, split, batchsize_per_replica, shuffle, transform, num_samples):
        super().__init__(split, batchsize_per_replica, shuffle, transform, num_samples)
        assert self._CIFAR_TYPE in [
            "cifar10",
            "cifar100",
        ], "CIFAR datasets only come in cifar10, cifar100"

        dataset = self._load_dataset()
        self.dataset = self.wrap_dataset(
            dataset,
            transform,
            batchsize_per_replica,
            shuffle=shuffle,
            subsample=num_samples,
        )

    @classmethod
    def from_config(cls, config):
        assert "split" in config
        split = config["split"]
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        default_transform = (
            _cifar_augment_transform()
            if split == "train"
            else _cifar_no_augment_transform()
        )
        transform = build_field_transform_default_imagenet(
            transform_config, default_transform=default_transform
        )
        return cls(split, batchsize_per_replica, shuffle, transform, num_samples)

    def _load_dataset(self):
        # set up CIFAR dataset:
        set_proxies()
        if self._CIFAR_TYPE == "cifar10":
            dataset = datasets.CIFAR10(
                DATA_PATH, train=(self.split == "train"), download=True
            )
        else:
            dataset = datasets.CIFAR100(
                DATA_PATH, train=(self.split == "train"), download=True
            )
        unset_proxies()
        self.transform = transforms.Compose(
            [TupleToMapTransform(["input", "target"]), self.transform]
        )
        return dataset


@register_dataset("cifar10")
class Cifar10Dataset(CifarDataset):
    _CIFAR_TYPE = "cifar10"


@register_dataset("cifar100")
class Cifar100Dataset(CifarDataset):
    _CIFAR_TYPE = "cifar100"
