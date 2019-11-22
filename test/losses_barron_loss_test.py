#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from classy_vision.losses import BarronLoss, build_loss


class TestBarronLoss(unittest.TestCase):
    def _get_config(self):
        return {"name": "barron", "size_average": True, "alpha": 1.0, "c": 1.0}

    def _get_outputs(self):
        return torch.tensor([[2.0]])

    def _get_targets(self):
        return torch.tensor([3.0])

    def test_build_barron(self):
        config = self._get_config()
        crit = build_loss(config)
        self.assertTrue(isinstance(crit, BarronLoss))
        self.assertEqual(crit.size_average, config["size_average"])
        self.assertAlmostEqual(crit.alpha, config["alpha"])
        self.assertAlmostEqual(crit.c, config["c"])

    def test_barron(self):
        config = self._get_config()
        crit = BarronLoss.from_config(config)
        outputs = self._get_outputs()
        targets = self._get_targets()
        self.assertAlmostEqual(crit(outputs, targets).item(), 0.41421353816986084)

        # Alpha = 0
        config = self._get_config()
        config["alpha"] = 0.0
        crit = BarronLoss.from_config(config)
        outputs = self._get_outputs()
        targets = self._get_targets()
        self.assertAlmostEqual(crit(outputs, targets).item(), 0.40546512603759766)

        # Alpha = inf
        config = self._get_config()
        config["alpha"] = float("inf")
        crit = BarronLoss.from_config(config)
        outputs = self._get_outputs()
        targets = self._get_targets()
        self.assertAlmostEqual(crit(outputs, targets).item(), 0.39346933364868164)

    def test_deep_copy(self):
        config = self._get_config()
        crit1 = build_loss(config)
        self.assertTrue(isinstance(crit1, BarronLoss))
        outputs = self._get_outputs()
        targets = self._get_targets()
        crit1(outputs, targets)

        crit2 = copy.deepcopy(crit1)
        self.assertAlmostEqual(
            crit1(outputs, targets).item(), crit2(outputs, targets).item()
        )
