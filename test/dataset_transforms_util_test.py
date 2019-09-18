#!/usr/bin/env python3

import unittest

import torch
import torchvision.transforms as transforms
from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
)
from classy_vision.dataset.transforms.util import (
    build_field_transform_default_imagenet,
    imagenet_no_augment_transform,
)


class DatasetTransformsUtilTest(unittest.TestCase):
    def get_test_image_dataset(self):
        config = {
            "crop_size": 224,
            "num_channels": 3,
            "num_classes": 10,
            "seed": 0,
            "class_ratio": 0.5,
            "num_samples": 100,
        }
        dataset = RandomImageBinaryClassDataset(config)
        return dataset

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
            torch.allclose(output_image, imagenet_no_augment_transform()(input_image))
        )
