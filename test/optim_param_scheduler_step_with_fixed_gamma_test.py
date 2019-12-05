#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import build_param_scheduler
from classy_vision.optim.param_scheduler.step_with_fixed_gamma_scheduler import (
    StepWithFixedGammaParamScheduler,
)


class TestStepWithFixedGammaScheduler(unittest.TestCase):
    _num_epochs = 12

    def _get_valid_config(self):
        return {
            "name": "step_with_fixed_gamma",
            "base_lr": 1,
            "gamma": 0.1,
            "num_decays": 3,
            "num_epochs": self._num_epochs,
        }

    def test_invalid_config(self):
        config = self._get_valid_config()

        # Invalid num epochs
        bad_config = copy.deepcopy(config)
        bad_config["num_epochs"] = -1
        with self.assertRaises(AssertionError):
            StepWithFixedGammaParamScheduler.from_config(bad_config)

        # Invalid num_decays
        bad_config["num_decays"] = 0
        with self.assertRaises(AssertionError):
            StepWithFixedGammaParamScheduler.from_config(bad_config)

        # Invalid base_lr
        bad_config = copy.deepcopy(config)
        bad_config["base_lr"] = -0.01
        with self.assertRaises(AssertionError):
            StepWithFixedGammaParamScheduler.from_config(bad_config)

        # Invalid gamma
        bad_config = copy.deepcopy(config)
        bad_config["gamma"] = [2]
        with self.assertRaises(AssertionError):
            StepWithFixedGammaParamScheduler.from_config(bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = StepWithFixedGammaParamScheduler.from_config(config)
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            1,
            1,
            1,
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.01,
            0.001,
            0.001,
            0.001,
        ]

        for param, expected_param in zip(schedule, expected_schedule):
            self.assertAlmostEqual(param, expected_param)

    def test_build_step_with_fixed_gamma_scheduler(self):
        config = self._get_valid_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, StepWithFixedGammaParamScheduler))
