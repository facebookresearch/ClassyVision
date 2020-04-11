#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from classy_vision.dataset.transforms.mixup import MixupTransform


class DatasetTransformsMixupTest(unittest.TestCase):
    def test_mixup_transform_single_label(self):
        alpha = 2.0
        num_classes = 3
        mixup_transform = MixupTransform(alpha, num_classes)
        sample = {
            "input": torch.rand(4, 3, 224, 224, dtype=torch.float32),
            "target": torch.as_tensor([0, 1, 2, 2], dtype=torch.int32),
        }
        sample_mixup = mixup_transform(sample)
        self.assertTrue(sample["input"].shape == sample_mixup["input"].shape)
        self.assertTrue(sample_mixup["target"].shape[0] == 4)
        self.assertTrue(sample_mixup["target"].shape[1] == 3)

    def test_mixup_transform_single_label_missing_num_classes(self):
        alpha = 2.0
        mixup_transform = MixupTransform(alpha, None)
        sample = {
            "input": torch.rand(4, 3, 224, 224, dtype=torch.float32),
            "target": torch.as_tensor([0, 1, 2, 2], dtype=torch.int32),
        }
        with self.assertRaises(Exception):
            mixup_transform(sample)

    def test_mixup_transform_multi_label(self):
        alpha = 2.0
        mixup_transform = MixupTransform(alpha, None)
        sample = {
            "input": torch.rand(4, 3, 224, 224, dtype=torch.float32),
            "target": torch.as_tensor(
                [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]],
                dtype=torch.int32,
            ),
        }
        sample_mixup = mixup_transform(sample)
        self.assertTrue(sample["input"].shape == sample_mixup["input"].shape)
        self.assertTrue(sample["target"].shape == sample_mixup["target"].shape)
