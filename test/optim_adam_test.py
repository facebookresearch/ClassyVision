#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.optim_test_util import TestOptimizer

from classy_vision.optim.adam import Adam


class TestAdamOptimizer(TestOptimizer, unittest.TestCase):
    def _check_momentum_buffer(self):
        return False

    def _get_config(self):
        return {
            "name": "adam",
            "num_epochs": 90,
            "lr": 0.1,
            "betas": (0.9, 0.99),
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "amsgrad": False,
        }

    def _instance_to_test(self):
        return Adam
