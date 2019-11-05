#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
)
from classy_vision.dataset.transforms.util import build_field_transform_default_imagenet


class LightingTransformTest(unittest.TestCase):
    def get_test_image_dataset(self):
        return RandomImageBinaryClassDataset(
            crop_size=224, class_ratio=0.5, num_samples=100, seed=0
        )

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
