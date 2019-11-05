#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torchvision.transforms as transforms

from . import register_dataset
from .classy_dataset import ClassyDataset
from .core import RandomImageBinaryClassDataset
from .transforms.util import ImagenetConstants, build_field_transform_default_imagenet


@register_dataset("synthetic_image")
class SyntheticImageClassificationDataset(ClassyDataset):
    @classmethod
    def get_available_splits(cls):
        return ["train", "val", "test"]

    def __init__(
        self,
        batchsize_per_replica,
        shuffle,
        transform,
        num_samples,
        crop_size,
        class_ratio,
        seed,
        split=None,
    ):
        super().__init__(split, batchsize_per_replica, shuffle, transform, num_samples)

        self.dataset = RandomImageBinaryClassDataset(
            crop_size, class_ratio, num_samples, seed
        )

    @classmethod
    def from_config(cls, config):
        assert all(key in config for key in ["crop_size", "class_ratio", "seed"])
        split = config.get("split")
        crop_size = config["crop_size"]
        class_ratio = config["class_ratio"]
        seed = config["seed"]
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        default_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=ImagenetConstants.MEAN, std=ImagenetConstants.STD
                ),
            ]
        )
        transform = build_field_transform_default_imagenet(
            transform_config, default_transform=default_transform
        )
        return cls(
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            crop_size,
            class_ratio,
            seed,
            split=split,
        )
