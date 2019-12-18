#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict, Optional, Union

from classy_vision.dataset import ClassyDataset, register_dataset
from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
    SampleType,
)
from classy_vision.dataset.transforms import ClassyTransform, build_transforms


@register_dataset("my_dataset")
class MyDataset(ClassyDataset):
    def __init__(
        self,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Union[ClassyTransform, Callable]],
        num_samples: int,
        crop_size: int,
        class_ratio: float,
        seed: int,
    ) -> None:
        dataset = RandomImageBinaryClassDataset(
            crop_size, class_ratio, num_samples, seed, SampleType.TUPLE
        )
        super().__init__(
            dataset, batchsize_per_replica, shuffle, transform, num_samples
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MyDataset":
        assert all(key in config for key in ["crop_size", "class_ratio", "seed"])

        crop_size = config["crop_size"]
        class_ratio = config["class_ratio"]
        seed = config["seed"]
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        transform = build_transforms(transform_config)
        return cls(
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            crop_size,
            class_ratio,
            seed,
        )
