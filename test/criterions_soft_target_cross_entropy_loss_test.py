#!/usr/bin/env python3

import copy
import unittest

import torch
from classy_vision.criterions import build_criterion
from classy_vision.criterions.soft_target_cross_entropy_loss import (
    SoftTargetCrossEntropyLoss,
)


class TestSoftTargetCrossEntropyLoss(unittest.TestCase):
    def _get_config(self):
        return {
            "name": "soft_target_cross_entropy",
            "ignore_index": -1,
            "reduction": "mean",
        }

    def _get_outputs(self):
        return torch.tensor([[1.0, 7.0, 0.0, 0.0, 2.0]])

    def _get_targets(self):
        return torch.tensor([[1, 0, 0, 0, 1]])

    def _get_loss(self):
        return 5.51097965

    def test_build_soft_target_cross_entropy(self):
        config = self._get_config()
        crit = build_criterion(config)
        self.assertTrue(isinstance(crit, SoftTargetCrossEntropyLoss))
        self.assertEqual(crit._ignore_index, -1)
        self.assertEqual(crit._reduction, "mean")

    def test_soft_target_cross_entropy(self):
        config = self._get_config()
        crit = SoftTargetCrossEntropyLoss(config)
        outputs = self._get_outputs()
        targets = self._get_targets()
        self.assertAlmostEqual(crit(outputs, targets).item(), self._get_loss())

        # Verify ignore index works
        outputs = self._get_outputs()
        targets = torch.tensor([[-1, 0, 0, 0, 1]])
        self.assertAlmostEqual(crit(outputs, targets).item(), 5.01097918)

    def test_unnormalized_soft_target_cross_entropy(self):
        config = {
            "name": "soft_target_cross_entropy",
            "ignore_index": -1,
            "reduction": "mean",
            "normalize_targets": None,
        }
        crit = SoftTargetCrossEntropyLoss(config)
        outputs = self._get_outputs()
        targets = self._get_targets()
        self.assertAlmostEqual(crit(outputs, targets).item(), 11.0219593)

        # Verify ignore index works
        outputs = self._get_outputs()
        targets = torch.tensor([[-1, 0, 0, 0, 1]])
        self.assertAlmostEqual(crit(outputs, targets).item(), 5.01097965)

    def test_ignore_row(self):
        # If a sample has no valid targets, it should be ignored in the reduction.
        config = self._get_config()
        crit = SoftTargetCrossEntropyLoss(config)
        outputs = torch.tensor([[1.0, 7.0, 0.0, 0.0, 2.0], [4.0, 2.0, 1.0, 6.0, 0.5]])
        targets = torch.tensor([[1, 0, 0, 0, 1], [-1, -1, -1, -1, -1]])
        self.assertAlmostEqual(crit(outputs, targets).item(), self._get_loss())

    def test_deep_copy(self):
        config = self._get_config()
        crit = build_criterion(config)
        self.assertTrue(isinstance(crit, SoftTargetCrossEntropyLoss))
        outputs = self._get_outputs()
        targets = self._get_targets()
        crit(outputs, targets)

        crit2 = copy.deepcopy(crit)
        self.assertAlmostEqual(crit2(outputs, targets).item(), self._get_loss())
