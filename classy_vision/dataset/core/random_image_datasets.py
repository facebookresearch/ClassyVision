#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image

from ...generic.util import numpy_seed
from .dataset import Dataset


class RandomImageDataset(Dataset):
    def __init__(self, crop_size, num_channels, num_classes, num_samples, seed):
        self.crop_size = crop_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.seed = seed

    def __getitem__(self, idx):
        with numpy_seed(self.seed + idx):
            return {
                "input": Image.fromarray(
                    (
                        np.random.standard_normal(
                            [self.crop_size, self.crop_size, self.num_channels]
                        )
                        * 255
                    ).astype(np.uint8)
                ),
                "target": np.random.randint(self.num_classes),
            }

    def __len__(self):
        return self.num_samples


class RandomImageBinaryClassDataset(Dataset):
    def __init__(self, crop_size, class_ratio, num_samples, seed):
        self.crop_size = crop_size
        # User Defined Class Imbalace Ratio
        self.class_ratio = class_ratio
        self.num_samples = num_samples
        self.seed = seed

    def __getitem__(self, idx):
        with numpy_seed(self.seed + idx):
            class_id = int(np.random.random() < self.class_ratio)
            image = np.zeros((self.crop_size, self.crop_size, 3))
            image[:, :, class_id] = np.random.random([self.crop_size, self.crop_size])
            image[:, :, 2] = np.random.random([self.crop_size, self.crop_size])
            return {
                "input": Image.fromarray((image * 255).astype(np.uint8)),
                "target": class_id,
            }

    def __len__(self):
        return self.num_samples
