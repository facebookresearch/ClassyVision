#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.optim.param_scheduler import build_param_scheduler
from classy_vision.optim.param_scheduler.constant_scheduler import (
    ConstantParamScheduler,
)


class TestFixedScheduler(unittest.TestCase):
    _num_epochs = 12

    def _get_valid_config(self):
        return {"name": "constant", "num_epochs": self._num_epochs, "value": 0.1}

    def test_invalid_config(self):
        # Invalid num epochs
        config = self._get_valid_config()

        bad_config = copy.deepcopy(config)
        del bad_config["value"]
        with self.assertRaises(AssertionError):
            ConstantParamScheduler.from_config(bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = ConstantParamScheduler.from_config(config)
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.assertEqual(schedule, expected_schedule)
        # The input for the scheduler should be in the interval [0;1), open
        with self.assertRaises(RuntimeError):
            scheduler(1)

    def test_build_constant_scheduler(self):
        config = self._get_valid_config()
        scheduler = build_param_scheduler(config)
        self.assertTrue(isinstance(scheduler, ConstantParamScheduler))
