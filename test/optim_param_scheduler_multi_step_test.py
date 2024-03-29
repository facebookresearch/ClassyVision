#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import (
    build_param_scheduler,
    MultiStepParamScheduler,
)


class TestMultiStepParamScheduler(unittest.TestCase):
    _num_epochs = 12

    def _get_valid_config(self):
        return {
            "name": "multistep",
            "num_epochs": self._num_epochs,
            "values": [0.1, 0.01, 0.001, 0.0001],
            "milestones": [4, 6, 8],
        }

    def test_invalid_config(self):
        # Invalid num epochs
        config = self._get_valid_config()

        bad_config = copy.deepcopy(config)
        bad_config["num_epochs"] = -1
        with self.assertRaises(ValueError):
            MultiStepParamScheduler.from_config(bad_config)

        # Invalid values
        bad_config["num_epochs"] = config["num_epochs"]
        del bad_config["values"]
        with self.assertRaises((AssertionError, TypeError)):
            MultiStepParamScheduler.from_config(bad_config)

        bad_config["values"] = {"a": "b"}
        with self.assertRaises(ValueError):
            MultiStepParamScheduler.from_config(bad_config)

        bad_config["values"] = []
        with self.assertRaises(ValueError):
            MultiStepParamScheduler.from_config(bad_config)

        # Invalid drop epochs
        bad_config["values"] = config["values"]
        bad_config["milestones"] = {"a": "b"}
        with self.assertRaises(ValueError):
            MultiStepParamScheduler.from_config(bad_config)

        # Too many
        bad_config["milestones"] = [3, 6, 8, 12]
        with self.assertRaises(ValueError):
            MultiStepParamScheduler.from_config(bad_config)

        # Too few
        bad_config["milestones"] = [3, 6]
        with self.assertRaises(ValueError):
            MultiStepParamScheduler.from_config(bad_config)

        # Exceeds num_epochs
        bad_config["milestones"] = [3, 6, 12]
        with self.assertRaises(ValueError):
            MultiStepParamScheduler.from_config(bad_config)

        # Out of order
        bad_config["milestones"] = [3, 8, 6]
        with self.assertRaises(ValueError):
            MultiStepParamScheduler.from_config(bad_config)

    def _test_config_scheduler(self, config, expected_schedule):
        scheduler = MultiStepParamScheduler.from_config(config)
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        self.assertEqual(schedule, expected_schedule)

    def test_scheduler(self):
        config = self._get_valid_config()
        expected_schedule = [
            0.1,
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.001,
            0.001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
        ]
        self._test_config_scheduler(config, expected_schedule)

    def test_default_config(self):
        config = self._get_valid_config()
        default_config = copy.deepcopy(config)
        # Default equispaced drop_epochs behavior
        del default_config["milestones"]
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
        self._test_config_scheduler(default_config, expected_schedule)

    def test_build_non_equi_step_scheduler(self):
        config = self._get_valid_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, MultiStepParamScheduler))
