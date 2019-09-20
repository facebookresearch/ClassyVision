#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import build_param_scheduler
from classy_vision.optim.param_scheduler.cosine_decay_scheduler import (
    CosineDecayParamScheduler,
)


class TestCosineScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_config(self):
        return {
            "name": "cosine",
            "num_epochs": self._num_epochs,
            "base_lr": 0.1,
            "min_lr": 0,
        }

    def test_invalid_config(self):
        # Invalid num epochs
        config = self._get_valid_config()

        bad_config = copy.deepcopy(config)
        # Invalid Base lr
        bad_config["num_epochs"] = config["num_epochs"]
        del bad_config["base_lr"]
        with self.assertRaises(AssertionError):
            CosineDecayParamScheduler(bad_config)

        # Invalid min_lr
        bad_config["base_lr"] = config["base_lr"]
        del bad_config["min_lr"]
        with self.assertRaises(AssertionError):
            CosineDecayParamScheduler(bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = CosineDecayParamScheduler(config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            0.1,
            0.0976,
            0.0905,
            0.0794,
            0.0655,
            0.05,
            0.0345,
            0.0206,
            0.0095,
            0.0024,
        ]

        self.assertEqual(schedule, expected_schedule)

    def test_build_cosine_scheduler(self):
        config = self._get_valid_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, CosineDecayParamScheduler))
