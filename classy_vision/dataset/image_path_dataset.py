#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .classy_dataset import ClassyDataset
from .core import ListDataset
from .transforms import build_transforms
from .transforms.classy_transform import ClassyTransform
from .transforms.util import TupleToMapTransform


def _load_dataset(image_paths, targets):
    if targets is None:
        targets = [torch.tensor([]) for _ in image_paths]
    if isinstance(image_paths, str):
        assert os.path.isdir(
            image_paths
        ), "Expect image_paths to be a dir when it is a string"
        dataset = datasets.ImageFolder(image_paths)
        preproc_transform = TupleToMapTransform(list_of_map_keys=["input", "target"])
    else:
        dataset = ListDataset(image_paths, targets)
        preproc_transform = None
    return dataset, preproc_transform


class ImagePathDataset(ClassyDataset):
    """Dataset which reads images from a local filesystem. Implements ClassyDataset.

    The image paths provided can be:
        - A single directory location, in which case the data is expected to be
            arranged in a format similar to :class:`torchvision.datasets.ImageFolder`.
            The targets will be inferred from the directory structure.
        - A list of paths, in which case the list will contain the paths to all the
            images. In this situation, the targets can be specified by the targets
            argument.
    """

    def __init__(
        self,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Union[ClassyTransform, Callable]],
        num_samples: Optional[int],
        image_paths: Union[str, List[str]],
        targets: Optional[List[Any]] = None,
    ):
        """Constructor for ImagePathDataset.

        Args:
            batchsize_per_replica: Positive integer indicating batch size for each
                replica
            shuffle: Whether we should shuffle between epochs
            transform: Transform to be applied to each sample
            num_samples: When set, this restricts the number of samples provided by
                the dataset
            image_paths: A directory or a list of file paths where images can be found.
            targets: If a list of file paths is specified, this argument can
                be used to specify a target for each path (must be same length
                as list of file paths). If no targets are needed or image_paths is
                a directory, then targets should be None.
        """
        # TODO(@mannatsingh): we should be able to call build_dataset() to create
        # datasets from this class.
        assert image_paths is not None, "image_paths needs to be provided"
        assert targets is None or isinstance(image_paths, list), (
            "targets cannot be specified when image_paths is a directory containing "
            "the targets in the directory structure"
        )
        dataset, preproc_transform = _load_dataset(image_paths, targets)
        super().__init__(
            dataset, batchsize_per_replica, shuffle, transform, num_samples
        )
        # Some of the base datasets from _load_dataset have different
        # sample formats, the preproc_transform should map them all to
        # the dict {"input": img, "target": label} format
        if preproc_transform is not None:
            self.transform = transforms.Compose([preproc_transform, self.transform])

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        image_paths: Union[str, List[str]],
        targets: Optional[List[Any]] = None,
    ):
        """Instantiates ImagePathDataset from a config.

        Because image_paths / targets can be arbitrarily long, we
        allow passing in the image paths and targets from python in
        addition to the configuration parameter.

        Args:
            config: A configuration for ImagePathDataset.
                See :func:`__init__` for parameters expected in the config.
            image_paths: Directory or list of image paths.
                See :func:`__init__` for more details
            targets: Optional list of targets for dataset.
                See :func:`__init__` for more details
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
            image_paths,
            targets=targets,
        )
