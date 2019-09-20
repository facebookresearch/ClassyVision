#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle

import numpy as np
from classy_vision.dataset.core import ListDataset

from . import register_dataset
from .classy_dataset import ClassyDataset
from .transforms.util import build_field_transform_default_imagenet


# constants for COCO dataset:
# COCO jsons are here '/mnt/vol/gfsai-east/ai-group/users/prigoyal/data/coco'
# train2014 has 82081 images which have annotations (out of 82783 images)
#   so train2014 has 702 images with no annotations
# val2014 has 40137 images which have annotations (out of 40504 images)
#   so val2014 has 367 images with no annotations
ANNOTATION_PATH = "/mnt/vol/gfsai-east/ai-group/users/imisra/datasets/coco"
IMAGE_PATH = "/data/local/packages/ai-group.coco_%s/prod/%s"
NUM_CLASSES = 80


def _get_annot_suffix(split):
    if split == "train":
        annot_suffix = "train2014"
    elif split == "val":
        annot_suffix = "val2014"
    elif split == "minival":
        annot_suffix = "minival2014"
    return annot_suffix


def _get_image_suffix(split):
    if split == "train":
        img_suffix = "train2014"
    elif split == "val":
        img_suffix = "val2014"
    elif split == "minival":
        img_suffix = "val2014"
    return img_suffix


def _get_image_dir(split):
    image_suffix = _get_image_suffix(split)
    image_dir = IMAGE_PATH % (image_suffix, "coco_" + image_suffix)
    return image_dir


# TODO: @imisra, COCO dataset paths seem not to be correct
@register_dataset("coco")
class CocoDataset(ClassyDataset):
    def __init__(self, config):
        super(CocoDataset, self).__init__(config)
        # For memoizing target names
        self._target_names = None
        self.dataset = self._load_dataset()
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = self.parse_config(self._config)
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
        annot_suffix = _get_annot_suffix(self._split)
        annot_file = os.path.join(
            ANNOTATION_PATH, "one_hot_labels_" + annot_suffix + ".pkl"
        )
        with open(annot_file, "rb") as f:
            annot_data = pickle.load(f)
        image_labels_np = annot_data["gt_labels"]
        image_names = annot_data["image_names"]
        image_dir = _get_image_dir(self._split)
        image_paths = [os.path.join(image_dir, x) for x in image_names]
        image_labels = [np.array(x, dtype=np.int32) for x in image_labels_np.tolist()]
        dataset = ListDataset(image_paths, image_labels)
        return dataset

    # returns names of classes for a split name
    def get_target_names(self, split=None):
        if self._target_names is not None:
            return self._target_names

        class_file = os.path.join(ANNOTATION_PATH, "classnames.txt")
        classes = []
        with open(class_file, "r") as fh:
            for line in fh:
                classes.append(line.strip())
        self._target_names = classes
        return self._target_names
