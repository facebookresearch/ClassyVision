#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import (
    build_param_scheduler,
    LinearParamScheduler,
)


class TestLienarScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_intermediate(self):
        return [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

    def _get_valid_config(self):
        return {"name": "linear", "start_value": 0.0, "end_value": 0.1}

    def test_invalid_config(self):
        config = self._get_valid_config()

        bad_config = copy.deepcopy(config)
        # No start lr
        del bad_config["start_value"]
        with self.assertRaises((AssertionError, TypeError)):
            LinearParamScheduler.from_config(bad_config)

        # No end lr
        bad_config["start_value"] = config["start_value"]
        del bad_config["end_value"]
        with self.assertRaises((AssertionError, TypeError)):
            LinearParamScheduler.from_config(bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        # Check as warmup
        scheduler = LinearParamScheduler.from_config(config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [config["start_value"]] + self._get_valid_intermediate()
        self.assertEqual(schedule, expected_schedule)

        # Check as decay
        tmp = config["start_value"]
        config["start_value"] = config["end_value"]
        config["end_value"] = tmp
        scheduler = LinearParamScheduler.from_config(config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [config["start_value"]] + list(
            reversed(self._get_valid_intermediate())
        )
        self.assertEqual(schedule, expected_schedule)

    def test_build_linear_scheduler(self):
        config = self._get_valid_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, LinearParamScheduler))
