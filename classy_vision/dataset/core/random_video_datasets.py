#!/usr/bin/env python3

import numpy as np
import torch

from ...generic.util import torch_seed
from .dataset import Dataset


class RandomVideoDataset(Dataset):
    def __init__(self, config):
        self.config = config
        # video config
        self.channels = config["num_channels"]
        self.num_frames = config["num_frames"]
        self.height = config["height"]
        self.width = config["width"]
        # audio config
        self.sample_rate = config["sample_rate"]
        # misc config
        self.num_classes = config["num_classes"]
        self.seed = config["seed"]

    def __getitem__(self, idx):
        with torch_seed(self.seed + idx):
            return {
                "input": {
                    "video": torch.randint(
                        0,
                        256,
                        (self.num_frames, self.height, self.width, self.channels),
                        dtype=torch.uint8,
                    ),
                    "audio": torch.rand((self.sample_rate, 1), dtype=torch.float),
                },
                "target": np.random.randint(self.num_classes),
            }

    def __len__(self):
        return self.config["num_samples"]
