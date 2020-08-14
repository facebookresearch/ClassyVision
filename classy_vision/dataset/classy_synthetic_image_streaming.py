#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torchvision.transforms as transforms
from classy_vision.dataset import register_dataset
from classy_vision.dataset.classy_dataset import ClassyDataset
from classy_vision.dataset.core import RandomImageBinaryClassDataset
from classy_vision.dataset.dataloader_async_gpu_wrapper import DataloaderAsyncGPUWrapper
from classy_vision.dataset.dataloader_limit_wrapper import DataloaderLimitWrapper
from classy_vision.dataset.transforms.util import (
    ImagenetConstants,
    build_field_transform_default_imagenet,
)


@register_dataset("synthetic_image_streaming")
class SyntheticImageStreamingDataset(ClassyDataset):
    """
    Synthetic image dataset that behaves like a streaming dataset.

    Requires a "num_samples" argument which decides the number of samples in the
    phase. Also takes an optional "length" input which sets the length of the
    dataset.
    """

    def __init__(
        self,
        batchsize_per_replica,
        shuffle,
        transform,
        num_samples,
        crop_size,
        class_ratio,
        seed,
        length=None,
        async_gpu_copy: bool = False,
    ):
        if length is None:
            # If length not provided, set to be same as num_samples
            length = num_samples

        dataset = RandomImageBinaryClassDataset(crop_size, class_ratio, length, seed)
        super().__init__(
            dataset, batchsize_per_replica, shuffle, transform, num_samples
        )
        self.async_gpu_copy = async_gpu_copy

    @classmethod
    def from_config(cls, config):
        # Parse the config
        assert all(key in config for key in ["crop_size", "class_ratio", "seed"])
        length = config.get("length")
        crop_size = config["crop_size"]
        class_ratio = config["class_ratio"]
        seed = config["seed"]
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        async_gpu_copy = config.get("async_gpu_copy", False)

        # Build the transforms
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
            length=length,
            async_gpu_copy=async_gpu_copy,
        )

    def iterator(self, *args, **kwargs):
        dataloader = DataloaderLimitWrapper(
            super().iterator(*args, **kwargs),
            self.num_samples // self.get_global_batchsize(),
        )

        if self.async_gpu_copy:
            dataloader = DataloaderAsyncGPUWrapper(dataloader)

        return dataloader
