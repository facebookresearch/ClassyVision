#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from classy_vision.dataset.core import ListDataset

from . import register_dataset
from .classy_dataset import ClassyDataset
from .transforms.util import build_field_transform_default_imagenet


# constants for CUB-2011 dataset:
CANDIDATE_PATHS = ["/mnt/fair-flash-east/cub2011.img"]
SPLIT_FOLDER = "/mnt/vol/gfsfblearner-oregon/users/dhruvm/cub"
SPLIT_FILES = {
    "train": os.path.join(SPLIT_FOLDER, "train_path_and_label.txt"),
    "test": os.path.join(SPLIT_FOLDER, "test_path_and_label.txt"),
}
CLASS_FILE = "/mnt/vol/gfsai-east/ai-group/users/imisra/datasets/cub2011/classes.txt"
NUM_CLASSES = 200


@register_dataset("cub2011")
class Cub2011Dataset(ClassyDataset):
    def __init__(self, split, batchsize_per_replica, shuffle, transform, num_samples):
        super().__init__(split, batchsize_per_replica, shuffle, transform, num_samples)
        # For memoizing target names
        self._target_names = None
        self.dataset = self._load_dataset()
        self.dataset = self.wrap_dataset(
            self.dataset,
            transform,
            batchsize_per_replica,
            shuffle=shuffle,
            subsample=num_samples,
        )

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
        # find location of images:
        img_dir = None
        for path in CANDIDATE_PATHS:
            if os.path.isdir(path):
                img_dir = path
                break
        assert img_dir is not None and os.path.isdir(
            img_dir
        ), "CUB-2011 folder not found"
        img_dir = os.path.join(img_dir, "CUB_200_2011")

        # load correct split:
        with open(SPLIT_FILES[self.split], "rt") as fread:
            lines = [l.strip().split(",") for l in fread.readlines()]
            imgs = [os.path.join(img_dir, x[0][62:]) for x in lines]
            targets = [int(x[1]) - 1 for x in lines]

        # return image dataset:
        return ListDataset(imgs, targets)

    def get_target_names(self):
        if self._target_names is not None:
            return self._target_names
        class_names = []
        with open(CLASS_FILE, "r") as fh:
            for line in fh:
                cls_id, cls_name = line.split()
                class_names.append(cls_name.strip())

        self._target_names = class_names
        return self._target_names
