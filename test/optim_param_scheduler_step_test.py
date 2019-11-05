#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import build_param_scheduler
from classy_vision.optim.param_scheduler.step_scheduler import StepParamScheduler


class TestStepScheduler(unittest.TestCase):
    _num_epochs = 12

    def _get_valid_config(self):
        return {
            "name": "step",
            "num_epochs": self._num_epochs,
            "values": [0.1, 0.01, 0.001, 0.0001],
        }

    def test_invalid_config(self):
        # Invalid num epochs
        config = self._get_valid_config()

        bad_config = copy.deepcopy(config)
        bad_config["num_epochs"] = -1
        with self.assertRaises(AssertionError):
            StepParamScheduler.from_config(bad_config)

        # Invalid Values
        bad_config["num_epochs"] = config["num_epochs"]
        del bad_config["values"]
        with self.assertRaises(AssertionError):
            StepParamScheduler.from_config(bad_config)

        bad_config["values"] = {"a": "b"}
        with self.assertRaises(AssertionError):
            StepParamScheduler.from_config(bad_config)

        bad_config["values"] = []
        with self.assertRaises(AssertionError):
            StepParamScheduler.from_config(bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = StepParamScheduler.from_config(config)
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.01,
            0.001,
            0.001,
            0.001,
            0.0001,
            0.0001,
            0.0001,
        ]

        self.assertEqual(schedule, expected_schedule)

    def test_build_step_scheduler(self):
        config = self._get_valid_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, StepParamScheduler))
