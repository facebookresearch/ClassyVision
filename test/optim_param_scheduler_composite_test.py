#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import build_param_scheduler
from classy_vision.optim.param_scheduler.composite_scheduler import (
    CompositeParamScheduler,
    IntervalScaling,
    UpdateInterval,
)


class TestCompositeScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_long_config(self):
        return {
            "name": "composite",
            "schedulers": [
                {"name": "constant", "value": 0.1},
                {"name": "constant", "value": 0.2},
                {"name": "constant", "value": 0.3},
                {"name": "constant", "value": 0.4},
            ],
            "lengths": [0.2, 0.4, 0.1, 0.3],
        }

    def _get_lengths_sum_less_one_config(self):
        return {
            "name": "composite",
            "schedulers": [
                {"name": "constant", "value": 0.1},
                {"name": "constant", "value": 0.2},
            ],
            "lengths": [0.7, 0.2999],
        }

    def _get_valid_mixed_config(self):
        return {
            "name": "composite",
            "schedulers": [
                {"name": "step", "values": [0.1, 0.2, 0.3, 0.4, 0.5], "num_epochs": 10},
                {"name": "cosine", "start_lr": 0.42, "end_lr": 0.0001},
            ],
            "lengths": [0.5, 0.5],
        }

    def _get_valid_linear_config(self):
        return {
            "name": "composite",
            "schedulers": [
                {"name": "linear", "start_lr": 0.0, "end_lr": 0.5},
                {"name": "linear", "start_lr": 0.5, "end_lr": 1.0},
            ],
            "lengths": [0.5, 0.5],
            "interval_scaling": ["rescaled", "rescaled"],
        }

    def test_invalid_config(self):
        config = self._get_valid_mixed_config()
        bad_config = copy.deepcopy(config)

        # No schedulers
        bad_config["schedulers"] = []
        bad_config["lengths"] = []
        with self.assertRaises(AssertionError):
            CompositeParamScheduler.from_config(bad_config)

        # Size of schedulers and lengths doesn't match
        bad_config["schedulers"] = copy.deepcopy(config["schedulers"])
        bad_config["lengths"] = copy.deepcopy(config["lengths"])
        bad_config["schedulers"].append(bad_config["schedulers"][-1])
        with self.assertRaises(AssertionError):
            CompositeParamScheduler.from_config(bad_config)

        # Sum of lengths < 1
        bad_config["schedulers"] = copy.deepcopy(config["schedulers"])
        bad_config["lengths"][-1] -= 0.1
        with self.assertRaises(AssertionError):
            CompositeParamScheduler.from_config(bad_config)

        # Sum of lengths > 1
        bad_config["lengths"] = copy.deepcopy(config["lengths"])
        bad_config["lengths"][-1] += 0.1
        with self.assertRaises(AssertionError):
            CompositeParamScheduler.from_config(bad_config)

        # Bad value for update_interval
        bad_config["lengths"] = copy.deepcopy(config["lengths"])
        bad_config["update_interval"] = "epochs"
        with self.assertRaises(AssertionError):
            CompositeParamScheduler.from_config(bad_config)

        # Bad value for composition_mode
        del bad_config["update_interval"]
        bad_config["interval_scaling"] = ["rescaled", "rescaleds"]
        with self.assertRaises(AssertionError):
            CompositeParamScheduler.from_config(bad_config)

        # Wrong number composition modes
        del bad_config["interval_scaling"]
        bad_config["interval_scaling"] = ["rescaled"]
        with self.assertRaises(AssertionError):
            CompositeParamScheduler.from_config(bad_config)

        # Missing required parameters
        del bad_config["interval_scaling"]
        bad_config["lengths"] = config["lengths"]
        del bad_config["lengths"]
        with self.assertRaises(AssertionError):
            CompositeParamScheduler.from_config(bad_config)

        bad_config["lengths"] = config["lengths"]
        del bad_config["schedulers"]
        with self.assertRaises(AssertionError):
            CompositeParamScheduler.from_config(bad_config)

    def test_long_scheduler(self):
        config = self._get_valid_long_config()

        scheduler = CompositeParamScheduler.from_config(config)
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4]

        self.assertEqual(schedule, expected_schedule)

    def test_scheduler_lengths_within_epsilon_of_one(self):
        config = self._get_lengths_sum_less_one_config()
        scheduler = CompositeParamScheduler.from_config(config)
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
        self.assertEqual(schedule, expected_schedule)

    def test_scheduler_update_interval(self):
        config = self._get_valid_mixed_config()

        # Check default
        scheduler = CompositeParamScheduler.from_config(config)
        self.assertEqual(scheduler.update_interval, UpdateInterval.STEP)

        # Check step
        step_config = copy.deepcopy(config)
        step_config["update_interval"] = "step"
        scheduler = build_param_scheduler(step_config)
        self.assertEqual(scheduler.update_interval, UpdateInterval.STEP)

        # Check epoch
        epoch_config = copy.deepcopy(config)
        epoch_config["update_interval"] = "epoch"
        scheduler = build_param_scheduler(epoch_config)
        self.assertEqual(scheduler.update_interval, UpdateInterval.EPOCH)

    def test_build_composite_scheduler(self):
        config = self._get_valid_mixed_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, CompositeParamScheduler))

        schedulers = [
            build_param_scheduler(scheduler_config)
            for scheduler_config in config["schedulers"]
        ]
        composite = CompositeParamScheduler(
            schedulers=schedulers,
            lengths=config["lengths"],
            update_interval=UpdateInterval.EPOCH,
            interval_scaling=[IntervalScaling.RESCALED, IntervalScaling.FIXED],
        )
        self.assertTrue(isinstance(composite, CompositeParamScheduler))

    def test_scheduler_with_mixed_types(self):
        config = self._get_valid_mixed_config()
        scheduler_0 = build_param_scheduler(config["schedulers"][0])
        scheduler_1 = build_param_scheduler(config["schedulers"][1])

        # Check scaled
        config["interval_scaling"] = ["rescaled", "rescaled"]
        scheduler = CompositeParamScheduler.from_config(config)
        scaled_schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            round(scheduler_0(epoch_num / self._num_epochs), 4)
            for epoch_num in range(0, self._num_epochs, 2)
        ] + [
            round(scheduler_1(epoch_num / self._num_epochs), 4)
            for epoch_num in range(0, self._num_epochs, 2)
        ]
        self.assertEqual(scaled_schedule, expected_schedule)

        # Check fixed
        config["interval_scaling"] = ["fixed", "fixed"]
        scheduler = CompositeParamScheduler.from_config(config)
        fixed_schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            round(scheduler_0(epoch_num / self._num_epochs), 4)
            for epoch_num in range(0, int(self._num_epochs / 2))
        ] + [
            round(scheduler_1(epoch_num / self._num_epochs), 4)
            for epoch_num in range(int(self._num_epochs / 2), self._num_epochs)
        ]
        self.assertEqual(fixed_schedule, expected_schedule)

        # Check that default is rescaled
        del config["interval_scaling"]
        scheduler = CompositeParamScheduler.from_config(config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        self.assertEqual(scaled_schedule, schedule)
        # Check warmup of rescaled then fixed
        config["interval_scaling"] = ["rescaled", "fixed"]
        scheduler = CompositeParamScheduler.from_config(config)
        fixed_schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            round(scheduler_0(epoch_num / self._num_epochs), 4)
            for epoch_num in range(0, int(self._num_epochs), 2)
        ] + [
            round(scheduler_1(epoch_num / self._num_epochs), 4)
            for epoch_num in range(int(self._num_epochs / 2), self._num_epochs)
        ]
        self.assertEqual(fixed_schedule, expected_schedule)

    def test_linear_scheduler_no_gaps(self):
        config = self._get_valid_linear_config()

        # Check rescaled
        scheduler = CompositeParamScheduler.from_config(config)
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.assertEqual(expected_schedule, schedule)

        # Check fixed composition gives same result as only 1 scheduler
        config["schedulers"][1] = config["schedulers"][0]
        config["interval_scaling"] = ["fixed", "fixed"]
        scheduler = CompositeParamScheduler.from_config(config)
        linear_scheduler = build_param_scheduler(config["schedulers"][0])
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            linear_scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        self.assertEqual(expected_schedule, schedule)
