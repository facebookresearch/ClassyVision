#!/usr/bin/env python3

import unittest

from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
)
from classy_vision.dataset.transforms.util import build_field_transform_default_imagenet


class LightingTransformTest(unittest.TestCase):
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

    def test_lighting_transform_no_errors(self):
        """
        Tests that the lighting transform runs without any errors.
        """
        dataset = self.get_test_image_dataset()

        config = [{"name": "ToTensor"}, {"name": "lighting"}]
        transform = build_field_transform_default_imagenet(config)
        sample = dataset[0]
        try:
            # test that lighting has been registered and runs without errors
            transform(sample)
        except Exception:
            self.fail("LightingTransform raised an exception")
        return
