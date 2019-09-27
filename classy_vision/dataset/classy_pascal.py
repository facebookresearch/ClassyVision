#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
from classy_vision.dataset.core import ListDataset

from . import register_dataset
from .classy_dataset import ClassyDataset
from .transforms.util import build_field_transform_default_imagenet


# constants for VOC datasets:
VOC2007_IMG_DIR = "/mnt/fair-flash-east/VOC2007"
VOC2012_IMG_DIR = "/mnt/fair-flash-east/VOC2012"
VOC2012_SPLITS_DIR = (
    "/mnt/vol/gfsai-east/ai-group/users/imisra/datasets/" "VOC2012/cls_splits"
)
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def _get_images_labels_singlelabel(image_dir, splits_dir, split):
    # loop over the correct split for all classes:
    imgs, tgts = [], []
    class_names = CLASSES
    for cls_num, class_name in enumerate(class_names):
        with open("%s/%s_%s.txt" % (splits_dir, class_name, split), "r") as fread:
            lines = fread.readlines()
            # add only positive images for this class:
            cur_imgs = [line.split(" ")[0] for line in lines]
            cur_tgts = [int(line.split(" ")[-1]) for line in lines]
            num = len(cur_imgs)
            imgs.extend([cur_imgs[n] for n in range(num) if cur_tgts[n] == 1])
            tgts.extend([cls_num for n in range(num) if cur_tgts[n] == 1])

    # return image dataset:
    img_paths = [os.path.join(image_dir, "JPEGImages/%s.jpg" % img) for img in imgs]
    return img_paths, tgts


# function to read the txt files in VOC
def _get_images_labels_multilabel(image_dir, splits_dir, split):
    cls_idx_map = {}
    class_names = CLASSES
    for idx, cls in enumerate(class_names):
        cls_idx_map[idx] = cls
    num_classes = len(CLASSES)
    # we will construct a map for image name to the vector of -1, 0, 1
    img_labels_map = {}
    for cls_num, class_name in enumerate(class_names):
        with open("%s/%s_%s.txt" % (splits_dir, class_name, split), "r") as fread:
            for line in fread:
                img_name, orig_label = line.strip().split()
                if img_name not in img_labels_map:
                    # save int32
                    img_labels_map[img_name] = -np.ones(num_classes, dtype=np.int32)
                orig_label = int(orig_label)
                # in VOC, not present, set it to 0 as train target
                if orig_label == -1:
                    orig_label = 0
                # in VOC, ignore, set it to -1 as train target
                elif orig_label == 0:
                    orig_label = -1
                img_labels_map[img_name][cls_num] = orig_label
    img_paths, img_labels = [], []
    for item in sorted(img_labels_map.keys()):
        img_paths.append(os.path.join(image_dir, "JPEGImages", item + ".jpg"))
        img_labels.append(img_labels_map[item])
    return img_paths, img_labels


class PascalDataset(ClassyDataset):
    def __init__(self, config, pascal_version, multilabel):
        super(PascalDataset, self).__init__(config)
        self._pascal_version = pascal_version
        self._multilabel = multilabel
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
        # assertions:
        assert self._pascal_version in ["voc2007", "voc2012"]
        image_dir = (
            VOC2007_IMG_DIR if self._pascal_version == "voc2007" else VOC2012_IMG_DIR
        )
        splits_dir = (
            os.path.join(VOC2007_IMG_DIR, "ImageSets/Main")
            if self._pascal_version == "voc2007"
            else VOC2012_SPLITS_DIR
        )
        if self._multilabel is False:
            img_paths, img_labels = _get_images_labels_singlelabel(
                image_dir, splits_dir, self._split
            )
        else:
            img_paths, img_labels = _get_images_labels_multilabel(
                image_dir, splits_dir, self._split
            )
        return ListDataset(img_paths, img_labels)

    def get_target_names(self):
        return CLASSES


@register_dataset("pascal_voc2007")
class PascalVOC2007Dataset(PascalDataset):
    def __init__(self, config):
        super(PascalVOC2007Dataset, self).__init__(config, "voc2007", False)


@register_dataset("pascal_voc2007_ml")
class PascalVOC2007MLDataset(PascalDataset):
    def __init__(self, config):
        super(PascalVOC2007MLDataset, self).__init__(config, "voc2007", True)


# TODO: @imisra, the 2012 datasets don't seem to work, potentially
# they no longer reference the correct directories
#
# @register_dataset('pascal_voc2012')
# class PascalVOC2012Dataset(PascalDataset):
#     def __init__(self, config):
#         super(PascalVOC2012Dataset, self).__init__(
#             config,
#             'voc2012',
#             False,
#         )


# @register_dataset('pascal_voc2012_ml')
# class PascalVOC2012MLDataset(PascalDataset):
#     def __init__(self, config):
#         super(PascalVOC2012MLDataset, self).__init__(
#             config,
#             'voc2012',
#             True,
#         )
