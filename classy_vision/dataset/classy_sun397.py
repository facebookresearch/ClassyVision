#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from classy_vision.dataset import register_dataset
from classy_vision.dataset.classy_dataset import ClassyDataset
from classy_vision.dataset.core import ListDataset

from .transforms.util import build_field_transform_default_imagenet


# constants for SUN397 dataset:
CANDIDATE_PATHS = ["/mnt/fair-flash-east/SUN397.img"]
META_DATA_DIR = "/mnt/vol/gfsai-east/ai-group/users/imisra/datasets/SUN397"
NUM_CLASSES = 397


@register_dataset("sun397")
class Sun397Dataset(ClassyDataset):
    def __init__(self, config):
        super(Sun397Dataset, self).__init__(config)
        # For memoizing target names
        self._target_names = None
        self.dataset = self._load_dataset()
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = self.parse_config(config)
        transform = build_field_transform_default_imagenet(
            transform_config, split=self._split
        )
        self.dataset = self.wrap_dataset(
            self.dataset,
            transform,
            batchsize_per_replica,
            shuffle=shuffle,
            subsample=num_samples,
        )

    def _load_dataset(self):
        # TODO(aadcock): allow access to predefined train / test splits
        # SUN397 defines 10 different train/test splits,
        # none of which cover the full 108754 images
        # each pre-defined train split is 19850 images each
        # each pre-defined test split is 19850 images each
        # we just define the "all" split that has the full 108754 images
        # find location of images
        img_dir = None
        for path in CANDIDATE_PATHS:
            if os.path.isdir(path):
                img_dir = path
                break

        assert img_dir is not None and os.path.isdir(
            img_dir
        ), "SUN397 folder: %s not found" % (img_dir)

        img_list_file = os.path.join(META_DATA_DIR, "all_image_list.txt")
        img_list = []
        with open(img_list_file, "r") as fh:
            for line in fh:
                img_list.append(line.strip())
        # WARNING: a lot of SUN images are TIFF/BMP/GIF/PNG files.
        # But every file has a .jpg extension
        img_paths = [os.path.join(img_dir, x) for x in img_list]
        class_names_orig = self.get_target_names()
        class_names_idx_map = {}
        for idx, cls in enumerate(class_names_orig):
            class_names_idx_map[cls] = idx

        tgts = []
        for i in range(len(img_list)):
            im = img_list[i]
            cls = "/".join(im.split("/")[:-1])
            idx = class_names_idx_map[cls]
            tgts.append(idx)

        # return image dataset:
        return ListDataset(img_paths, tgts)

    def get_target_names(self):
        # Memoize target names
        if self._target_names is not None:
            return self._target_names

        labels_file = os.path.join(META_DATA_DIR, "classnames_sanitized.txt")
        with open(labels_file, "r") as fh:
            class_names = []
            for line in fh:
                class_name = line.strip()
                class_names.append(class_name)
        assert len(class_names) == NUM_CLASSES
        self._target_names = class_names
        return self._target_names
