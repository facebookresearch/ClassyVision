#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
)
from classy_vision.dataset.transforms.autoaugment import ImagenetAutoAugment  # noqa
from classy_vision.dataset.transforms.util import build_field_transform_default_imagenet


class AutoaugmentTransformTest(unittest.TestCase):
    def get_test_image_dataset(self):
        return RandomImageBinaryClassDataset(
            crop_size=224, class_ratio=0.5, num_samples=100, seed=0
        )

    def test_imagenet_autoaugment_transform_no_errors(self):
        """
        Tests that the imagenet autoaugment transform runs without any errors.
        """
        dataset = self.get_test_image_dataset()

        config = [{"name": "imagenet_autoaugment"}]
        transform = build_field_transform_default_imagenet(config)
        sample = dataset[0]
        # test that imagenet autoaugment has been registered and runs without errors
        transform(sample)
