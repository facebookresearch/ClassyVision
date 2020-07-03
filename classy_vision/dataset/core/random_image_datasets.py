#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import numpy as np
from PIL import Image

from ...generic.util import numpy_seed


class SampleType(Enum):
    DICT = "dict"
    TUPLE = "tuple"
    LIST = "list"


def _get_typed_sample(input, target, id, sample_type):
    if sample_type == SampleType.DICT:
        return {"input": input, "target": target, "id": id}
    elif sample_type == SampleType.TUPLE:
        return (input, target)
    elif sample_type == SampleType.LIST:
        return [input, target]
    else:
        raise TypeError("Provided sample_type is not dict, list, tuple")


class RandomImageDataset:
    def __init__(
        self,
        crop_size,
        num_channels,
        num_classes,
        num_samples,
        seed,
        sample_type=SampleType.DICT,
    ):
        self.crop_size = crop_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.seed = seed
        self.sample_type = sample_type

    def __getitem__(self, idx):
        with numpy_seed(self.seed + idx):
            input = Image.fromarray(
                (
                    np.random.standard_normal(
                        [self.crop_size, self.crop_size, self.num_channels]
                    )
                    * 255
                ).astype(np.uint8)
            )
            target = np.random.randint(self.num_classes)
            return _get_typed_sample(input, target, idx, self.sample_type)

    def __len__(self):
        return self.num_samples


class RandomImageBinaryClassDataset:
    def __init__(
        self, crop_size, class_ratio, num_samples, seed, sample_type=SampleType.DICT
    ):
        self.crop_size = crop_size
        # User Defined Class Imbalace Ratio
        self.class_ratio = class_ratio
        self.num_samples = num_samples
        self.seed = seed
        self.sample_type = sample_type

    def __getitem__(self, idx):
        with numpy_seed(self.seed + idx):
            class_id = int(np.random.random() < self.class_ratio)
            image = np.zeros((self.crop_size, self.crop_size, 3))
            image[:, :, class_id] = np.random.random([self.crop_size, self.crop_size])
            image[:, :, 2] = np.random.random([self.crop_size, self.crop_size])
            input = Image.fromarray((image * 255).astype(np.uint8))
            target = class_id
            return _get_typed_sample(input, target, idx, self.sample_type)

    def __len__(self):
        return self.num_samples
