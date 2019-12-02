#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Union

from classy_vision.dataset import ClassyDataset, register_dataset
from classy_vision.dataset.transforms import ClassyTransform, build_transforms
from torchvision.datasets.imagenet import ImageNet


@register_dataset("classy_imagenet")
class ImageNetDataset(ClassyDataset):
    def __init__(
        self,
        split: Optional[str],
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Union[ClassyTransform, Callable]],
        num_samples: Optional[int],
        root: str,
        download: bool = None,
    ):
        dataset = ImageNet(root=root, split=split, download=download)
        super().__init__(
            dataset, split, batchsize_per_replica, shuffle, transform, num_samples
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImageNetDataset":
        """Instantiates a ImageNetDataset from a configuration.

        Args:
            config: A configuration for a ImageNetDataset.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A ImageNetDataset instance.
        """
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        split = config.get("split")
        root = config.get("root")
        download = config.get("download")

        transform = build_transforms(transform_config)
        return cls(
            split=split,
            batchsize_per_replica=batchsize_per_replica,
            shuffle=shuffle,
            transform=transform,
            num_samples=num_samples,
            root=root,
            download=download,
        )
