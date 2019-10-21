#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from classy_vision.dataset import register_dataset
from classy_vision.dataset.classy_dataset import ClassyDataset

from .transforms.util import TupleToMapTransform, build_field_transform_default_imagenet


# constants for the OmniGlot dataset:
DATA_PATH = "/mnt/fair-flash-east/omniglot.img"


# find classes in dataset folder with two levels of labels:
def _find_classes(self, folder):

    # find classes and super-classes:
    classes = []
    super_classes = [
        f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))
    ]
    for super_class in super_classes:
        subfolder = os.path.join(folder, super_class)
        classes.extend(
            [
                os.path.join(subfolder, f)
                for f in os.listdir(subfolder)
                if os.path.isdir(os.path.join(subfolder, f))
            ]
        )

    # sort classes and make dictionary:
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# make dataset in folder with two levels of labels:
def _make_dataset(_dir, class_to_idx, _extensions):
    images = []
    for folder, target in class_to_idx.items():
        for _, __, fnames in sorted(os.walk(folder)):
            for fname in sorted(fnames):
                if datasets.folder.is_image_file(fname):
                    path = os.path.join(folder, fname)
                    item = (path, target)
                    images.append(item)
    return images


@register_dataset("omniglot")
class OmniglotDataset(ClassyDataset):
    def __init__(self, split, batchsize_per_replica, shuffle, transform, num_samples):
        super().__init__(split, batchsize_per_replica, shuffle, transform, num_samples)
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
        # monkey-patch ImageFolder:
        orig_find_classes = datasets.folder.DatasetFolder._find_classes
        orig_make_dataset = datasets.folder.make_dataset
        datasets.folder.DatasetFolder._find_classes = _find_classes
        datasets.folder.make_dataset = _make_dataset

        # return dataset:
        img_dir = os.path.join(
            DATA_PATH,
            "images_background" if self.split == "train" else "images_evaluation",
        )
        dataset = datasets.ImageFolder(img_dir)

        # undo monkey-patch and return:
        datasets.folder.DatasetFolder._find_classes = orig_find_classes
        datasets.folder.make_dataset = orig_make_dataset

        self.transform = transforms.Compose(
            [TupleToMapTransform(["input", "target"]), self.transform]
        )
        self._num_classes = len(dataset.classes)
        return dataset
