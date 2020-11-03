import unittest
from test.generic.optim_test_util import TestOptimizer

from classy_vision.optim.adam import Adadelta


class TestAdadeltaOptimizer(TestOptimizer, unittest.Testcase):
    def _check_momentum_buffer(self):
        return False

    def _get_config(self):
        return {
            "name": "adadelta",
            "num_epochs": 90,
            "lr": 0.1,
            "rho": 0.9,
            "eps": 1e-6,
            "weight_decay": 0
        }

    def _instance_to_test(self):
        return Adadelta
