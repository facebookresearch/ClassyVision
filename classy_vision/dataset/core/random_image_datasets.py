#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image

from ...generic.util import numpy_seed
from .dataset import Dataset


class RandomImageDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.size = config["crop_size"]
        self.channels = config["num_channels"]
        self.num_classes = config["num_classes"]
        self.seed = config["seed"]

    def __getitem__(self, idx):
        with numpy_seed(self.seed + idx):
            return {
                "input": Image.fromarray(
                    (
                        np.random.standard_normal([self.size, self.size, self.channels])
                        * 255
                    ).astype(np.uint8)
                ),
                "target": np.random.randint(self.num_classes),
            }

    def __len__(self):
        return self.config["num_samples"]


class RandomImageBinaryClassDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.size = config["crop_size"]
        self.seed = config["seed"]
        # User Defined Class Imbalace Ratio
        self.ratio = config["class_ratio"]

    def __getitem__(self, idx):
        with numpy_seed(self.seed + idx):
            class_id = int(np.random.random() < self.ratio)
            image = np.zeros((self.size, self.size, 3))
            image[:, :, class_id] = np.random.random([self.size, self.size])
            image[:, :, 2] = np.random.random([self.size, self.size])
            return {
                "input": Image.fromarray((image * 255).astype(np.uint8)),
                "target": class_id,
            }

    def __len__(self):
        return self.config["num_samples"]
