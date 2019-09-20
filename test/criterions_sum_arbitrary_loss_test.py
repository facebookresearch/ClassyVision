#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from classy_vision.criterions import (
    ClassyCriterion,
    build_criterion,
    register_criterion,
)
from classy_vision.criterions.sum_arbitrary_loss import SumArbitraryLoss


@register_criterion("mock_a")
class MockCriterion1(ClassyCriterion):
    def forward(self, pred, target):
        return torch.tensor(1.0)


@register_criterion("mock_b")
class MockCriterion2(ClassyCriterion):
    def forward(self, pred, target):
        return torch.tensor(2.0)


@register_criterion("mock_c")
class MockCriterion3(ClassyCriterion):
    def forward(self, pred, target):
        return torch.tensor(3.0)


class TestSumArbitraryLoss(unittest.TestCase):
    def _get_config(self):
        return {
            "name": "sum_arbitrary",
            "weights": [1.0, 1.0, 1.0],
            "losses": [{"name": "mock_a"}, {"name": "mock_b"}, {"name": "mock_c"}],
        }

    def _get_outputs(self):
        return torch.tensor([[2.0, 8.0]])

    def _get_targets(self):
        return torch.tensor([1])

    def test_build_sum_arbitrary(self):
        config = self._get_config()
        crit = build_criterion(config)
        self.assertTrue(isinstance(crit, SumArbitraryLoss))
        self.assertAlmostEqual(crit.weights, [1.0, 1.0, 1.0])
        mod_list = [MockCriterion1, MockCriterion2, MockCriterion3]
        for idx, crit_type in enumerate(mod_list):
            self.assertTrue(isinstance(crit.losses[idx], crit_type))

    def test_sum_arbitrary(self):
        config = self._get_config()
        crit = SumArbitraryLoss(config)
        outputs = self._get_outputs()
        targets = self._get_targets()
        self.assertAlmostEqual(crit(outputs, targets).item(), 1.0 + 2.0 + 3.0)

        # Verify changing losses works
        new_config = config.copy()
        new_config.update(
            {"losses": [{"name": "mock_a"}, {"name": "mock_b"}], "weights": [1.0, 1.0]}
        )
        crit = SumArbitraryLoss(new_config)
        self.assertAlmostEqual(crit(outputs, targets).item(), 1.0 + 2.0)

        # Verify changing weights works
        new_config = config.copy()
        new_config.update({"weights": [1.0, 2.0, 3.0]})
        crit = SumArbitraryLoss(new_config)
        self.assertAlmostEqual(
            crit(outputs, targets).item(), 1.0 + 2.0 * 2.0 + 3.0 * 3.0
        )

    def test_deep_copy(self):
        config = self._get_config()
        crit1 = build_criterion(config)
        self.assertTrue(isinstance(crit1, SumArbitraryLoss))
        outputs = self._get_outputs()
        targets = self._get_targets()
        crit1(outputs, targets)

        crit2 = copy.deepcopy(crit1)
        self.assertAlmostEqual(
            crit1(outputs, targets).item(), crit2(outputs, targets).item()
        )
