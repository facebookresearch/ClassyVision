#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from classy_vision.dataset import register_dataset
from classy_vision.dataset.classy_dataset import ClassyDataset
from classy_vision.generic.util import set_proxies, unset_proxies

from .transforms.util import TupleToMapTransform, build_field_transform_default_imagenet


# constants for the SVHN datasets:
DATA_PATH = "/mnt/vol/gfsai-flash-east/ai-group/users/lvdmaaten/svhn"
NUM_CLASSES = 10


@register_dataset("svhn")
class SVHNDataset(ClassyDataset):
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
        set_proxies()
        dataset = datasets.SVHN(DATA_PATH, split=self.split, download=True)
        unset_proxies()
        self.transform = transforms.Compose(
            [TupleToMapTransform(["input", "target"]), self.transform]
        )
        return dataset
