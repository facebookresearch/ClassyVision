#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from classy_vision.losses import LabelSmoothingCrossEntropyLoss, build_loss


class TestLabelSmoothingCrossEntropyLoss(unittest.TestCase):
    def test_build_label_smoothing_cross_entropy(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.1,
        }
        crit = build_loss(config)
        self.assertTrue(isinstance(crit, LabelSmoothingCrossEntropyLoss))
        self.assertEqual(crit._ignore_index, -1)

    def test_smoothing_one_hot_targets(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.1,
        }
        crit = build_loss(config)
        targets = torch.tensor([[0, 0, 0, 0, 1]])
        self.assertTrue(isinstance(crit, LabelSmoothingCrossEntropyLoss))
        valid_targets = crit.compute_valid_targets(targets, 5)
        self.assertTrue(
            torch.allclose(valid_targets, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]))
        )
        smoothed_targets = crit.smooth_targets(valid_targets, 5)
        self.assertTrue(
            torch.allclose(
                smoothed_targets,
                torch.tensor([[0.2 / 11, 0.2 / 11, 0.2 / 11, 0.2 / 11, 10.2 / 11]]),
            )
        )

    def test_smoothing_ignore_index_one_hot_targets(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.5,
        }
        crit = build_loss(config)
        targets = torch.tensor([[-1, 0, 0, 0, 1]])
        self.assertTrue(isinstance(crit, LabelSmoothingCrossEntropyLoss))
        valid_targets = crit.compute_valid_targets(targets, 5)
        self.assertTrue(
            torch.allclose(valid_targets, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]))
        )
        smoothed_targets = crit.smooth_targets(valid_targets, 5)
        self.assertTrue(
            torch.allclose(
                smoothed_targets,
                torch.tensor([[1 / 15, 1 / 15, 1 / 15, 1 / 15, 11 / 15]]),
            )
        )

    def test_smoothing_multilabel_one_hot_targets(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.5,
        }
        crit = build_loss(config)
        targets = torch.tensor([[1, 0, 0, 0, 1]])
        self.assertTrue(isinstance(crit, LabelSmoothingCrossEntropyLoss))
        valid_targets = crit.compute_valid_targets(targets, 5)
        self.assertTrue(
            torch.allclose(valid_targets, torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]]))
        )

        smoothed_targets = crit.smooth_targets(valid_targets, 5)
        self.assertTrue(
            torch.allclose(
                smoothed_targets,
                torch.tensor([[6 / 15, 1 / 15, 1 / 15, 1 / 15, 6 / 15]]),
            )
        )

    def test_smoothing_all_ones_one_hot_targets(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.1,
        }
        crit = build_loss(config)
        targets = torch.tensor([[1, 1, 1, 1]])
        self.assertTrue(isinstance(crit, LabelSmoothingCrossEntropyLoss))
        valid_targets = crit.compute_valid_targets(targets, 4)
        self.assertTrue(
            torch.allclose(valid_targets, torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
        )

        smoothed_targets = crit.smooth_targets(valid_targets, 4)
        self.assertTrue(
            torch.allclose(smoothed_targets, torch.tensor([[0.25, 0.25, 0.25, 0.25]]))
        )

    def test_smoothing_mixed_one_hot_targets(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.5,
        }
        crit = build_loss(config)
        targets = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1]])
        self.assertTrue(isinstance(crit, LabelSmoothingCrossEntropyLoss))
        valid_targets = crit.compute_valid_targets(targets, 5)
        self.assertTrue(
            torch.allclose(
                valid_targets,
                torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0]]),
            )
        )
        smoothed_targets = crit.smooth_targets(valid_targets, 5)
        self.assertTrue(
            torch.allclose(
                smoothed_targets,
                torch.tensor(
                    [
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [6 / 15, 1 / 15, 1 / 15, 1 / 15, 6 / 15],
                    ]
                ),
            )
        )

    def test_smoothing_class_targets(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.5,
        }
        crit = build_loss(config)
        targets = torch.tensor([4, -1])
        self.assertTrue(isinstance(crit, LabelSmoothingCrossEntropyLoss))
        valid_targets = crit.compute_valid_targets(targets, 5)
        self.assertTrue(
            torch.allclose(
                valid_targets,
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
            )
        )
        smoothed_targets = crit.smooth_targets(valid_targets, 5)
        self.assertTrue(
            torch.allclose(
                smoothed_targets,
                torch.tensor(
                    [
                        [1 / 15, 1 / 15, 1 / 15, 1 / 15, 11 / 15],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                    ]
                ),
            )
        )

    def test_unnormalized_label_smoothing_cross_entropy(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.5,
        }
        crit = LabelSmoothingCrossEntropyLoss.from_config(config)
        outputs = torch.tensor([[0.0, 7.0, 0.0, 0.0, 2.0]])
        targets = torch.tensor([[0, 0, 0, 0, 1]])
        self.assertAlmostEqual(crit(outputs, targets).item(), 5.07609558)

    def test_ignore_index_label_smoothing_cross_entropy(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.2,
        }
        crit = LabelSmoothingCrossEntropyLoss.from_config(config)
        outputs = torch.tensor([[0.0, 7.0]])
        targets = torch.tensor([[-1]])
        self.assertAlmostEqual(crit(outputs, targets).item(), 3.50090909)

    def test_class_integer_label_smoothing_cross_entropy(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.2,
        }
        crit = LabelSmoothingCrossEntropyLoss.from_config(config)
        outputs = torch.tensor([[1.0, 2.0], [0.0, 2.0]])
        targets = torch.tensor([[0], [1]])
        self.assertAlmostEqual(crit(outputs, targets).item(), 0.76176142)

    def test_deep_copy(self):
        config = {
            "name": "label_smoothing_cross_entropy",
            "ignore_index": -1,
            "smoothing_param": 0.5,
        }
        crit = build_loss(config)
        self.assertTrue(isinstance(crit, LabelSmoothingCrossEntropyLoss))
        outputs = torch.tensor([[0.0, 7.0, 0.0, 0.0, 2.0]])
        targets = torch.tensor([[0, 0, 0, 0, 1]])
        crit(outputs, targets)

        crit2 = copy.deepcopy(crit)
        self.assertAlmostEqual(crit2(outputs, targets).item(), 5.07609558)
