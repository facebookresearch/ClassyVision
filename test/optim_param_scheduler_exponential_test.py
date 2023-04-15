#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import (
    build_param_scheduler,
    ExponentialParamScheduler,
)


class TestExponentialScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_config(self):
        return {"name": "exponential", "start_value": 2.0, "decay": 0.1}

    def _get_valid_intermediate_values(self):
        return [1.5887, 1.2619, 1.0024, 0.7962, 0.6325, 0.5024, 0.3991, 0.3170, 0.2518]

    def test_invalid_config(self):
        config = self._get_valid_config()

        bad_config = copy.deepcopy(config)
        # Invalid Base lr
        del bad_config["start_value"]
        with self.assertRaises((AssertionError, TypeError)):
            ExponentialParamScheduler.from_config(bad_config)

        # Invalid decay
        bad_config["start_value"] = config["start_value"]
        del bad_config["decay"]
        with self.assertRaises((AssertionError, TypeError)):
            ExponentialParamScheduler.from_config(bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = ExponentialParamScheduler.from_config(config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            config["start_value"]
        ] + self._get_valid_intermediate_values()

        self.assertEqual(schedule, expected_schedule)

    def test_build_exponential_scheduler(self):
        config = self._get_valid_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, ExponentialParamScheduler))
