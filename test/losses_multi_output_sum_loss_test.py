#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.losses import (
    ClassyLoss,
    MultiOutputSumLoss,
    build_loss,
    register_loss,
)


@register_loss("mock_1")
class MockLoss1(ClassyLoss):
    def forward(self, pred, target):
        return torch.tensor(1.0)

    @classmethod
    def from_config(cls, config):
        return cls()


class TestMultiOutputSumLoss(unittest.TestCase):
    def test_multi_output_sum_loss(self):
        config = {"name": "multi_output_sum_loss", "loss": {"name": "mock_1"}}
        crit = build_loss(config)
        self.assertTrue(isinstance(crit, MultiOutputSumLoss))

        # test with a single output
        output = torch.tensor([1.0, 2.3])
        target = torch.tensor(1.0)
        self.assertAlmostEqual(crit(output, target).item(), 1.0)

        # test with a list of outputs
        output = [torch.tensor([1.2, 3.2])] * 5
        target = torch.tensor(2.3)
        self.assertAlmostEqual(crit(output, target).item(), 5.0)
