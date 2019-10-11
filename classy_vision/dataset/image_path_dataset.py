#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path

import torchvision.datasets as datasets

from .classy_dataset import ClassyDataset
from .core import ListDataset, WrapDataset
from .transforms.util import build_field_transform_default_imagenet


def _transform_sample(sample):
    return {"input": sample["input"][0], "target": sample["input"][1]}


class ImagePathDataset(ClassyDataset):
    """
    A ClassyDataset class which reads images from image paths.

    image_paths: Can be
        - A single directory location, in which case the data is expected to be
            arranged in a format similar to torchvision.datasets.ImageFolder.
            The targets will be inferred from the directory structure.
        - A list of paths, in which case the list will contain the paths to all the
            images. In this situation, the targets can be specified by the targets
            argument.
    targets (optional): A list containing the target classes for each image
    """

    def __init__(
        self,
        batchsize_per_replica,
        shuffle,
        transform,
        num_samples,
        image_paths,
        targets=None,
        split=None,
    ):
        # TODO(@mannatsingh): we should be able to call build_dataset() to create
        # datasets from this class.
        assert image_paths is not None, "image_paths needs to be provided"
        assert targets is None or isinstance(image_paths, list), (
            "targets cannot be specified when image_paths is a directory containing "
            "the targets in the directory structure"
        )
        super().__init__(split, batchsize_per_replica, shuffle, transform, num_samples)

        dataset = self._load_dataset(image_paths, targets)
        self.dataset = self.wrap_dataset(
            dataset,
            transform,
            batchsize_per_replica=batchsize_per_replica,
            shuffle=shuffle,
            subsample=num_samples,
        )

    @classmethod
    def from_config(cls, config, image_paths, targets=None, default_transform=None):
        split = config.get("split")
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        transform = build_field_transform_default_imagenet(
            transform_config, default_transform=default_transform, split=split
        )
        return cls(
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            image_paths,
            targets=targets,
            split=split,
        )

    def _load_dataset(self, image_paths, targets):
        if isinstance(image_paths, str):
            assert os.path.isdir(
                image_paths
            ), "Expect image_paths to be a dir when it is a string"
            dataset = datasets.ImageFolder(image_paths)
            dataset = WrapDataset(dataset)
            # Wrap dataset places whole sample in input field by default
            # Remap this to input / targets since default wrapper does not
            # know which tensors are targets vs inputs
            dataset = dataset.transform(_transform_sample)
        else:
            dataset = ListDataset(image_paths, targets)
        return dataset
