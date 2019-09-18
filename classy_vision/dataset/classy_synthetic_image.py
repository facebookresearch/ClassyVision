#!/usr/bin/env python3


import torchvision.transforms as transforms

from . import register_dataset
from .classy_dataset import ClassyDataset
from .core import RandomImageBinaryClassDataset
from .transforms.util import ImagenetConstants, build_field_transform_default_imagenet


@register_dataset("synthetic_image")
class SyntheticImageClassificationDataset(ClassyDataset):
    @classmethod
    def get_available_splits(cls):
        return ["train", "val", "test"]

    def __init__(self, config):
        super(SyntheticImageClassificationDataset, self).__init__(config)
        self.dataset = RandomImageBinaryClassDataset(config)

        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = self.parse_config(self._config)
        default_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=ImagenetConstants.MEAN, std=ImagenetConstants.STD
                ),
            ]
        )
        transform = build_field_transform_default_imagenet(
            transform_config, default_transform=default_transform
        )
        self.dataset = self.wrap_dataset(
            self.dataset,
            transform,
            batchsize_per_replica,
            shuffle=shuffle,
            subsample=num_samples,
        )
