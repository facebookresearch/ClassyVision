#!/usr/bin/env python3

import copy
import unittest

import torch
from classy_vision.criterions import build_criterion
from classy_vision.criterions.sum_bce_with_logits_loss import SumBCEWithLogitsLoss
from classy_vision.criterions.sum_cross_entropy_loss import SumCrossEntropyLoss


class TestCriterion(unittest.TestCase):
    def setUp(self):
        self.cfg_sum_bce_with_logits = {
            "cls": SumBCEWithLogitsLoss,
            "config": {
                "name": "sum_bce_with_logits",
                "weight": torch.tensor([1.0, 1.0]),
                "reduction": "mean",
            },
            "outputs": [torch.tensor([0.999, 0.999]), torch.tensor([0.999, 0.999])],
            "targets": torch.tensor([1.0, 1.0]),
            "loss": 0.6270614545214017,
            "loss2": 1.2541229724884033,
        }
        self.cfg_sum_cross_entropy = {
            "cls": SumCrossEntropyLoss,
            "config": {
                "name": "sum_cross_entropy",
                "weight": torch.tensor([1.0, 1.0]),
                "ignore_index": -1,
                "reduction": "mean",
            },
            "outputs": [torch.tensor([[9.0, 1.0]]), torch.tensor([[2.0, 8.0]])],
            "targets": torch.tensor([1]),
            "loss": 8.002811431884766,
            "loss2": 8.002811431884766,
        }
        self.configs = [self.cfg_sum_bce_with_logits, self.cfg_sum_cross_entropy]

    def test_build_criterion(self):
        for cfg in self.configs:
            crit = build_criterion(cfg["config"])
            self.assertTrue(isinstance(crit, cfg["cls"]))
            self.assertAlmostEqual(crit._weight.numpy().tolist(), [1.0, 1.0])

    def test_criterion(self):
        for cfg in self.configs:
            crit = cfg["cls"](cfg["config"])
            self.assertAlmostEqual(
                crit(cfg["outputs"], cfg["targets"]).item(), cfg["loss"]
            )

    def test_deep_copy(self):
        for cfg in self.configs:
            crit = build_criterion(cfg["config"])
            crit2 = copy.deepcopy(crit)
            self.assertAlmostEqual(
                crit(cfg["outputs"], cfg["targets"]),
                crit2(cfg["outputs"], cfg["targets"]),
            )

    def test_sum_cross_entropy(self):
        # Verify ignore index works
        crit = SumCrossEntropyLoss(self.cfg_sum_cross_entropy["config"])
        self.assertAlmostEqual(
            crit(self.cfg_sum_cross_entropy["outputs"], torch.tensor([-1])).item(), 0.0
        )
