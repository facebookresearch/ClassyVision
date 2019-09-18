#!/usr/bin/env python3

import unittest

import torch
from classy_vision.criterions import (
    ClassyCriterion,
    build_criterion,
    register_criterion,
)
from classy_vision.criterions.multi_output_sum_loss import MultiOutputSumLoss


@register_criterion("mock_1")
class MockCriterion1(ClassyCriterion):
    def forward(self, pred, target):
        return torch.tensor(1.0)


class TestMultiOutputSumLoss(unittest.TestCase):
    def test_multi_output_sum_loss(self):
        config = {"name": "multi_output_sum_loss", "loss": {"name": "mock_1"}}
        crit = build_criterion(config)
        self.assertTrue(isinstance(crit, MultiOutputSumLoss))

        # test with a single output
        output = torch.tensor([1.0, 2.3])
        target = torch.tensor(1.0)
        self.assertAlmostEqual(crit(output, target).item(), 1.0)

        # test with a list of outputs
        output = [torch.tensor([1.2, 3.2])] * 5
        target = torch.tensor(2.3)
        self.assertAlmostEqual(crit(output, target).item(), 5.0)
