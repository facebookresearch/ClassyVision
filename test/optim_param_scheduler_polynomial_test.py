#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import build_param_scheduler
from classy_vision.optim.param_scheduler.polynomial_decay_scheduler import (
    PolynomialDecayParamScheduler,
)


class TestPolynomialScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_config(self):
        return {
            "name": "polynomial",
            "num_epochs": self._num_epochs,
            "base_lr": 0.1,
            "power": 1,
        }

    def test_invalid_config(self):
        # Invalid num epochs
        config = self._get_valid_config()

        # Invalid Base lr
        bad_config = copy.deepcopy(config)
        del bad_config["base_lr"]
        with self.assertRaises(AssertionError):
            PolynomialDecayParamScheduler.from_config(bad_config)

        # Invalid Power
        bad_config = copy.deepcopy(config)
        del bad_config["power"]
        with self.assertRaises(AssertionError):
            PolynomialDecayParamScheduler.from_config(bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = PolynomialDecayParamScheduler.from_config(config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 2)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

        self.assertEqual(schedule, expected_schedule)

    def test_build_polynomial_scheduler(self):
        config = self._get_valid_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, PolynomialDecayParamScheduler))
