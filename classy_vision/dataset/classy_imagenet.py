#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from . import register_dataset
from .classy_dataset import ClassyDataset
from .transforms.util import TupleToMapTransform, build_field_transform_default_imagenet


# constants for ImageNet dataset:
CANDIDATE_PATHS = [
    "/data/local/packages/ai-group.imagenet-full-size/prod/imagenet_full_size",
    "/mnt/fair-flash3-east/imagenet_full_size",
]
NUM_CLASSES = 1000


@register_dataset("imagenet")
class ImagenetDataset(ClassyDataset):
    @classmethod
    def get_available_splits(cls):
        return ["train", "val"]

    def __init__(self, split, batchsize_per_replica, shuffle, transform, num_samples):
        super().__init__(split, batchsize_per_replica, shuffle, transform, num_samples)
        # For memoizing target names
        self._target_names = None
        self.dataset = self._load_dataset()

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
        transform = build_field_transform_default_imagenet(
            transform_config, split=split
        )
        return cls(split, batchsize_per_replica, shuffle, transform, num_samples)

    def _load_dataset(self):
        # find location of images
        img_dir = None
        for path in CANDIDATE_PATHS:
            if os.path.isdir(path):
                img_dir = os.path.join(path, self.split)
                break
        assert img_dir is not None and os.path.isdir(
            img_dir
        ), "imagenet folder not found"

        dataset = datasets.ImageFolder(img_dir)
        self.transform = transforms.Compose(
            [TupleToMapTransform(["input", "target"]), self.transform]
        )
        return dataset

    # Imagenet dataset specific functions
    def get_target_names(self):
        # Memoize target names
        if self._target_names is not None:
            return self._target_names

        for path in CANDIDATE_PATHS:
            if os.path.isdir(path):
                break
        labels_file = os.path.join(path, "labels.txt")
        with open(labels_file, "r") as fh:
            class_names = []
            synset_names = []
            for line in fh:
                synset_name, class_name = line.strip().split(",")
                synset_names.append(synset_name)
                class_names.append(class_name)
        assert len(synset_names) == NUM_CLASSES
        assert len(class_names) == NUM_CLASSES
        self._target_names = (synset_names, class_names)
        return self._target_names
