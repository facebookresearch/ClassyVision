#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, List, Optional

import torchvision.transforms as transforms

from . import register_dataset
from .classy_dataset import ClassyDataset
from .core import RandomImageBinaryClassDataset
from .transforms import build_transforms
from .transforms.util import ImagenetConstants, build_field_transform_default_imagenet


@register_dataset("synthetic_image")
class SyntheticImageDataset(ClassyDataset):
    """Classy Dataset which produces random synthetic images with binary targets.

    The underlying dataset sets targets based on the channels in the image, so users can
    validate their setup by checking if they can get 100% accuracy on this dataset.
    Useful for testing since the dataset is much faster to initialize and fetch samples
    from, compared to real world datasets.
    """

    def __init__(
        self,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Callable],
        num_samples: int,
        crop_size: int,
        class_ratio: float,
        seed: int,
    ) -> None:
        """
        Args:
            batchsize_per_replica: Positive integer indicating batch size for each
                replica
            shuffle: Whether we should shuffle between epochs
            transform: When specified, transform to be applied to each sample
            num_samples: Number of samples to return
            crop_size: Image size, used for both height and width
            class_ratio: Ratio of the distribution of target classes
            seed: Seed used for image generation. Use the same seed to generate the same
                set of samples.
            split: When specified, split of dataset to use
        """
        dataset = RandomImageBinaryClassDataset(
            crop_size, class_ratio, num_samples, seed
        )
        super().__init__(
            dataset, batchsize_per_replica, shuffle, transform, num_samples
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SyntheticImageDataset":
        """Instantiates a SyntheticImageDataset from a configuration.

        Args:
            config: A configuration for a SyntheticImageDataset.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SyntheticImageDataset instance.
        """
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

        try:
            transform = build_transforms(transform_config)
        except Exception:
            logging.error(
                "We recently changed transform behavior"
                " do you need to update your config?"
                " See resnet50_synthetic_image_classy_config.json"
                " as an example."
            )
            raise

        return cls(
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            crop_size,
            class_ratio,
            seed,
        )
