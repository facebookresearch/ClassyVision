#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
import unittest

import numpy
import torch
import torchvision.transforms as transforms
from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
    SampleType,
)
from classy_vision.dataset.transforms import build_transforms
from classy_vision.dataset.transforms.util import (
    GenericImageTransform,
    ImagenetAugmentTransform,
    ImagenetNoAugmentTransform,
    build_field_transform_default_imagenet,
)


class DatasetTransformsUtilTest(unittest.TestCase):
    def get_test_image_dataset(self, sample_type):
        return RandomImageBinaryClassDataset(
            crop_size=224,
            class_ratio=0.5,
            num_samples=100,
            seed=0,
            sample_type=sample_type,
        )

    def transform_checks(
        self, sample, transform, expected_transform, key, transformed_key=None
    ):
        # If transformed key is None, then use key
        transformed_key = transformed_key if transformed_key is not None else key
        input_image = copy.deepcopy(sample[key])

        torch.manual_seed(0)
        numpy.random.seed(0)
        random.seed(0)
        output_image = transform(sample)[transformed_key]

        torch.manual_seed(0)
        numpy.random.seed(0)
        random.seed(0)
        self.assertTrue(torch.allclose(output_image, expected_transform(input_image)))

    def test_build_dict_field_transform_default_imagenet(self):
        dataset = self.get_test_image_dataset(SampleType.DICT)

        # should apply the transform in the config
        config = [{"name": "ToTensor"}]
        default_transform = transforms.Compose(
            [transforms.CenterCrop(100), transforms.ToTensor()]
        )
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform
        )
        sample = dataset[0]
        self.transform_checks(sample, transform, transforms.ToTensor(), "input")

        # should apply default_transform
        config = None
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform
        )
        sample = dataset[0]
        self.transform_checks(sample, transform, default_transform, "input")

        # should apply the transform for a test split
        transform = build_field_transform_default_imagenet(config, split="test")
        sample = dataset[0]
        self.transform_checks(sample, transform, ImagenetNoAugmentTransform(), "input")

    def test_build_tuple_field_transform_default_imagenet(self):
        dataset = self.get_test_image_dataset(SampleType.TUPLE)

        # should apply the transform in the config
        config = [{"name": "ToTensor"}]
        default_transform = transforms.Compose(
            [transforms.CenterCrop(100), transforms.ToTensor()]
        )
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform, key=0, key_map_transform=None
        )
        sample = dataset[0]
        self.transform_checks(sample, transform, transforms.ToTensor(), 0)

        # should apply default_transform
        config = None
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform, key=0, key_map_transform=None
        )
        sample = dataset[0]
        self.transform_checks(sample, transform, default_transform, 0)

        # should apply the transform for a test split
        transform = build_field_transform_default_imagenet(
            config, split="test", key=0, key_map_transform=None
        )
        sample = dataset[0]
        self.transform_checks(sample, transform, ImagenetNoAugmentTransform(), 0)

    def test_apply_transform_to_key_from_config(self):
        dataset = self.get_test_image_dataset(SampleType.DICT)

        config = [
            {
                "name": "apply_transform_to_key",
                "transforms": [{"name": "ToTensor"}],
                "key": "input",
            }
        ]
        transform = build_transforms(config)
        sample = dataset[0]
        self.transform_checks(sample, transform, transforms.ToTensor(), "input")

    def test_generic_image_transform(self):
        dataset = self.get_test_image_dataset(SampleType.TUPLE)

        # Check class constructor
        transform = GenericImageTransform(transform=transforms.ToTensor())
        sample = dataset[0]
        self.transform_checks(sample, transform, transforms.ToTensor(), 0, "input")

        transform = GenericImageTransform(split="train")
        sample = dataset[0]
        self.transform_checks(sample, transform, ImagenetAugmentTransform(), 0, "input")

        transform = GenericImageTransform(split="test")
        sample = dataset[0]
        self.transform_checks(
            sample, transform, ImagenetNoAugmentTransform(), 0, "input"
        )

        # Check from_config constructor / registry
        config = [
            {"name": "generic_image_transform", "transforms": [{"name": "ToTensor"}]}
        ]
        transform = build_transforms(config)
        sample = dataset[0]
        self.transform_checks(sample, transform, transforms.ToTensor(), 0, "input")

        # Check with Imagenet defaults
        config = [{"name": "generic_image_transform", "split": "train"}]
        transform = build_transforms(config)
        sample = dataset[0]
        self.transform_checks(sample, transform, ImagenetAugmentTransform(), 0, "input")

        config = [{"name": "generic_image_transform", "split": "test"}]
        transform = build_transforms(config)
        sample = dataset[0]
        self.transform_checks(
            sample, transform, ImagenetNoAugmentTransform(), 0, "input"
        )
