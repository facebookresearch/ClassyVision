#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
from typing import Any, Callable, Dict, List, Optional, Union

from torchvision import datasets, transforms

from . import ClassyDataset, register_dataset
from .core import ListDataset
from .transforms import ClassyTransform, TupleToMapTransform, build_transforms


def _is_torchvision_imagefolder(image_folder):
    with os.scandir(image_folder) as folder_iter:
        try:
            dir_entry = next(folder_iter)
            return dir_entry.is_dir()
        except StopIteration:
            raise OSError(f"Image folder {image_folder} is empty")


def _get_image_paths(image_folder):
    return [f"{image_folder}/{file}" for file in os.listdir(image_folder)]


def _load_dataset(image_folder, image_files):
    if image_folder is not None:
        if _is_torchvision_imagefolder(image_folder):
            return (
                datasets.ImageFolder(image_folder),
                TupleToMapTransform(list_of_map_keys=["input", "target"]),
            )
        else:
            image_files = _get_image_paths(image_folder)
    return ListDataset(image_files, metadata=None), None


@register_dataset("image_path")
class ImagePathDataset(ClassyDataset):
    """Dataset which reads images from a local filesystem. Implements ClassyDataset."""

    def __init__(
        self,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Union[ClassyTransform, Callable]] = None,
        num_samples: Optional[int] = None,
        image_folder: Optional[str] = None,
        image_files: Optional[List[str]] = None,
    ):
        """Constructor for ImagePathDataset.

        Only one of image_folder or image_files should be passed to specify the images.

        Args:
            batchsize_per_replica: Positive integer indicating batch size for each
                replica
            shuffle: Whether we should shuffle between epochs
            transform: Transform to be applied to each sample
            num_samples: When set, this restricts the number of samples provided by
                the dataset
            image_folder: A directory with one of the following structures -
                - A directory containing sub-directories with images for each target,
                    which is the format expected by
                    :class:`torchvision.datasets.ImageFolder` -

                    dog/xxx.png
                    dog/xxy.png
                    cat/123.png
                    cat/nsdf3.png

                    In this case, the targets are inferred from the sub-directories.
                - A directory containing images -

                    123.png
                    xyz.png

                    In this case, the targets are not returned (useful for inference).
            image_files: A list of image files -

                [
                    "123.png",
                    "dog/xyz.png",
                    "/home/cat/aaa.png"
                ]

                In this case, the targets are not returned (useful for inference).
        """
        if (image_folder is None) == (image_files is None):
            raise ValueError("One of image_folder and image_files should be provided")
        dataset, preproc_transform = _load_dataset(image_folder, image_files)
        super().__init__(
            dataset, batchsize_per_replica, shuffle, transform, num_samples
        )
        # Some of the base datasets from _load_dataset have different
        # sample formats, the preproc_transform should map them all to
        # the dict {"input": img, "target": label} format
        if preproc_transform is not None:
            self.transform = transforms.Compose([preproc_transform, self.transform])

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Instantiates ImagePathDataset from a config.

        Args:
            config: A configuration for ImagePathDataset.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            An ImagePathDataset instance.
        """
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
            image_folder=config.get("image_folder"),
            image_files=config.get("image_files"),
        )
