#!/usr/bin/env python3

import torchvision.datasets as datasets
from classy_vision.dataset import register_dataset
from classy_vision.dataset.classy_dataset import ClassyDataset
from classy_vision.dataset.core import WrapDataset
from classy_vision.generic.util import set_proxies, unset_proxies

from .transforms.util import build_field_transform_default_imagenet


# constants for the SVHN datasets:
DATA_PATH = "/mnt/vol/gfsai-flash-east/ai-group/users/lvdmaaten/svhn"
NUM_CLASSES = 10


@register_dataset("svhn")
class SVHNDataset(ClassyDataset):
    def __init__(self, config):
        super(SVHNDataset, self).__init__(config)
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
        set_proxies()
        dataset = datasets.SVHN(DATA_PATH, split=self._split, download=True)
        unset_proxies()
        dataset = WrapDataset(dataset)
        dataset = dataset.transform(
            lambda x: {"input": x["input"][0], "target": x["input"][1]}
        )
        return dataset
