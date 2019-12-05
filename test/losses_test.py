#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.losses import build_loss


class CriterionsTest(unittest.TestCase):
    """
    Test that build_transform is able to build torch losses correctly.
    """

    def _test_loss(self, config, output, target, expected_loss):
        # test that we are able to build losses from torch.nn.modules.loss
        # and that they work correctly

        crit = build_loss(config)

        # test that the weights are set correctly
        self.assertAlmostEqual(crit.weight.numpy().tolist(), [1.0, 1.0])

        # test that the loss is computed correctly
        self.assertAlmostEqual(crit(output, target).item(), expected_loss)

        # verify ignore index works
        if "ignore_index" in config:
            self.assertAlmostEqual(crit(output, torch.tensor([-1])).item(), 0.0)

    def test_cross_entropy_loss(self):
        """
        Test CrossEntropyLoss
        """
        config = {
            "name": "CrossEntropyLoss",
            "weight": [1.0, 1.0],
            "ignore_index": -1,
            "reduction": "mean",
        }
        output = torch.tensor([[9.0, 1.0]])
        target = torch.tensor([1])
        expected_loss = 8.000335693359375
        self._test_loss(config, output, target, expected_loss)

    def test_bce_with_logits_loss(self):
        """
        Test BCEWithLogitsLoss
        """
        config = {
            "name": "BCEWithLogitsLoss",
            "weight": [1.0, 1.0],
            "reduction": "mean",
        }
        output = torch.tensor([0.999, 0.999])
        target = torch.tensor([1.0, 1.0])
        expected_loss = 0.313530727260701
        self._test_loss(config, output, target, expected_loss)
