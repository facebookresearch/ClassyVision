#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import build_param_scheduler
from classy_vision.optim.param_scheduler.cosine_scheduler import CosineParamScheduler


class TestCosineScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_decay_config(self):
        return {"name": "cosine", "start_lr": 0.1, "end_lr": 0}

    def _get_valid_decay_config_intermediate_values(self):
        return [0.0976, 0.0905, 0.0794, 0.0655, 0.05, 0.0345, 0.0206, 0.0095, 0.0024]

    def test_invalid_config(self):
        # Invalid num epochs
        config = self._get_valid_decay_config()

        bad_config = copy.deepcopy(config)
        # Invalid Base lr
        del bad_config["start_lr"]
        with self.assertRaises(AssertionError):
            CosineParamScheduler.from_config(bad_config)

        # Invalid end_lr
        bad_config["start_lr"] = config["start_lr"]
        del bad_config["end_lr"]
        with self.assertRaises(AssertionError):
            CosineParamScheduler.from_config(bad_config)

    def test_scheduler_as_decay(self):
        config = self._get_valid_decay_config()

        scheduler = CosineParamScheduler.from_config(config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            config["start_lr"]
        ] + self._get_valid_decay_config_intermediate_values()

        self.assertEqual(schedule, expected_schedule)

    def test_scheduler_as_warmup(self):
        config = self._get_valid_decay_config()
        # Swap start and end lr to change to warmup
        tmp = config["start_lr"]
        config["start_lr"] = config["end_lr"]
        config["end_lr"] = tmp

        scheduler = CosineParamScheduler.from_config(config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        # Schedule should be decay reversed
        expected_schedule = [config["start_lr"]] + list(
            reversed(self._get_valid_decay_config_intermediate_values())
        )

        self.assertEqual(schedule, expected_schedule)

    def test_scheduler_warmup_decay_match(self):
        decay_config = self._get_valid_decay_config()
        decay_scheduler = CosineParamScheduler.from_config(decay_config)

        warmup_config = copy.deepcopy(decay_config)
        # Swap start and end lr to change to warmup
        tmp = warmup_config["start_lr"]
        warmup_config["start_lr"] = warmup_config["end_lr"]
        warmup_config["end_lr"] = tmp
        warmup_scheduler = CosineParamScheduler.from_config(warmup_config)

        decay_schedule = [
            round(decay_scheduler(epoch_num / 1000), 8) for epoch_num in range(1, 1000)
        ]
        warmup_schedule = [
            round(warmup_scheduler(epoch_num / 1000), 8) for epoch_num in range(1, 1000)
        ]

        self.assertEqual(decay_schedule, list(reversed(warmup_schedule)))

    def test_build_cosine_scheduler(self):
        config = self._get_valid_decay_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, CosineParamScheduler))
