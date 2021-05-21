#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from classy_vision.dataset.transforms.mixup import MixupTransform


class DatasetTransformsMixupTest(unittest.TestCase):
    def test_mixup_transform_single_label_image_batch(self):
        mixup_alpha = 2.0
        num_classes = 3

        for mode in ["batch", "pair", "elem"]:
            mixup_transform = MixupTransform(mixup_alpha, num_classes, mode=mode)
            sample = {
                "input": torch.rand(4, 3, 224, 224, dtype=torch.float32),
                "target": torch.as_tensor([0, 1, 2, 2], dtype=torch.int32),
            }
            sample_mixup = mixup_transform(sample)
            self.assertTrue(sample["input"].shape == sample_mixup["input"].shape)
            self.assertTrue(sample_mixup["target"].shape[0] == 4)
            self.assertTrue(sample_mixup["target"].shape[1] == 3)

    def test_cutmix_transform_single_label_image_batch(self):
        mixup_alpha = 0
        cutmix_alpha = 0.2
        num_classes = 3

        for mode in ["batch", "pair", "elem"]:
            cutmix_transform = MixupTransform(
                mixup_alpha,
                num_classes,
                cutmix_alpha=cutmix_alpha,
                mode=mode,
            )
            sample = {
                "input": torch.rand(4, 3, 224, 224, dtype=torch.float32),
                "target": torch.as_tensor([0, 1, 2, 2], dtype=torch.int32),
            }
            sample_cutmix = cutmix_transform(sample)
            self.assertTrue(sample["input"].shape == sample_cutmix["input"].shape)
            self.assertTrue(sample_cutmix["target"].shape[0] == 4)
            self.assertTrue(sample_cutmix["target"].shape[1] == 3)

    def test_mixup_cutmix_transform_single_label_image_batch(self):
        mixup_alpha = 0.3
        cutmix_alpha = 0.2
        num_classes = 3

        for mode in ["batch", "pair", "elem"]:
            cutmix_transform = MixupTransform(
                mixup_alpha,
                num_classes,
                cutmix_alpha=cutmix_alpha,
                switch_prob=0.5,
                mode=mode,
            )

            for _i in range(4):
                sample = {
                    "input": torch.rand(4, 3, 224, 224, dtype=torch.float32),
                    "target": torch.as_tensor([0, 1, 2, 2], dtype=torch.int32),
                }
                sample_cutmix = cutmix_transform(sample)
                self.assertTrue(sample["input"].shape == sample_cutmix["input"].shape)
                self.assertTrue(sample_cutmix["target"].shape[0] == 4)
                self.assertTrue(sample_cutmix["target"].shape[1] == 3)

    def test_mixup_cutmix_transform_single_label_image_batch_label_smooth(self):
        mixup_alpha = 0.3
        cutmix_alpha = 0.2
        num_classes = 3

        for mode in ["batch", "pair", "elem"]:
            cutmix_transform = MixupTransform(
                mixup_alpha,
                num_classes,
                cutmix_alpha=cutmix_alpha,
                switch_prob=0.5,
                mode=mode,
                label_smoothing=0.1,
            )

            for _i in range(4):
                sample = {
                    "input": torch.rand(4, 3, 224, 224, dtype=torch.float32),
                    "target": torch.as_tensor([0, 1, 2, 2], dtype=torch.int32),
                }
                sample_cutmix = cutmix_transform(sample)
                self.assertTrue(sample["input"].shape == sample_cutmix["input"].shape)
                self.assertTrue(sample_cutmix["target"].shape[0] == 4)
                self.assertTrue(sample_cutmix["target"].shape[1] == 3)

    def test_mixup_transform_single_label_image_batch_missing_num_classes(self):
        mixup_alpha = 2.0
        mixup_transform = MixupTransform(mixup_alpha, None)
        sample = {
            "input": torch.rand(4, 3, 224, 224, dtype=torch.float32),
            "target": torch.as_tensor([0, 1, 2, 2], dtype=torch.int32),
        }
        with self.assertRaises(Exception):
            mixup_transform(sample)

    def test_mixup_transform_multi_label_image_batch(self):
        mixup_alpha = 2.0
        mixup_transform = MixupTransform(mixup_alpha, None)
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

    def test_mixup_transform_single_label_multi_modal_batch(self):
        mixup_alpha = 2.0
        num_classes = 3
        mixup_transform = MixupTransform(mixup_alpha, num_classes)
        sample = {
            "input": {
                "video": torch.rand(4, 3, 4, 224, 224, dtype=torch.float32),
                "audio": torch.rand(4, 1, 40, 100, dtype=torch.float32),
            },
            "target": torch.as_tensor([0, 1, 2, 2], dtype=torch.int32),
        }
        mixup_transform(sample)

    def test_mixup_transform_multi_label_multi_modal_batch(self):
        mixup_alpha = 2.0
        mixup_transform = MixupTransform(mixup_alpha, None)
        sample = {
            "input": {
                "video": torch.rand(4, 3, 4, 224, 224, dtype=torch.float32),
                "audio": torch.rand(4, 1, 40, 100, dtype=torch.float32),
            },
            "target": torch.as_tensor(
                [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]],
                dtype=torch.int32,
            ),
        }
        mixup_transform(sample)
