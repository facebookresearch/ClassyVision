#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision.transforms as transforms
from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
)
from classy_vision.dataset.transforms.util import (
    ImagenetNoAugmentTransform,
    build_field_transform_default_imagenet,
)


class DatasetTransformsUtilTest(unittest.TestCase):
    def get_test_image_dataset(self):
        return RandomImageBinaryClassDataset(
            crop_size=224, class_ratio=0.5, num_samples=100, seed=0
        )

    def test_build_field_transform_default_imagenet(self):
        dataset = self.get_test_image_dataset()

        # should apply the transform in the config
        config = [{"name": "ToTensor"}]
        default_transform = transforms.Compose(
            [transforms.CenterCrop(100), transforms.ToTensor()]
        )
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform
        )
        sample = dataset[0]
        input_image = dataset[0]["input"]
        output_image = transform(sample)["input"]
        self.assertTrue(
            torch.allclose(output_image, transforms.ToTensor()(input_image))
        )

        # should apply default_transform
        config = None
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform
        )
        sample = dataset[0]
        input_image = dataset[0]["input"]
        output_image = transform(sample)["input"]
        self.assertTrue(torch.allclose(output_image, default_transform(input_image)))

        # should apply the transform for a test split
        transform = build_field_transform_default_imagenet(config, split="test")
        sample = dataset[0]
        input_image = dataset[0]["input"]
        output_image = transform(sample)["input"]
        self.assertTrue(
            torch.allclose(output_image, ImagenetNoAugmentTransform()(input_image))
        )
