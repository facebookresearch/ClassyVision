#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest
import unittest.mock as mock

import torch
import torch.nn as nn
from classy_vision.hooks import ExponentialMovingAverageModelHook
from classy_vision.models import ClassyModel


class TestModel(ClassyModel):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10)

    def init_fc_weight(self):
        nn.init.zeros_(self.fc.weight)

    def update_fc_weight(self):
        nn.init.ones_(self.fc.weight)

    def forward(self, x):
        return self.bn(self.fc(x))


class TestExponentialMovingAverageModelHook(unittest.TestCase):
    def _map_device_string(self, device):
        return "cuda" if device == "gpu" else "cpu"

    def _test_exponential_moving_average_hook(self, model_device, hook_device):
        task = mock.MagicMock()
        model = TestModel().to(device=self._map_device_string(model_device))
        local_variables = {}
        task.base_model = model
        task.train = True
        decay = 0.5
        num_updates = 10
        model.init_fc_weight()
        exponential_moving_average_hook = ExponentialMovingAverageModelHook(
            decay=decay, device=hook_device
        )

        exponential_moving_average_hook.on_start(task, local_variables)
        exponential_moving_average_hook.on_phase_start(task, local_variables)
        # set the weights to all ones and simulate 10 updates
        task.base_model.update_fc_weight()
        fc_weight = model.fc.weight.clone()
        for _ in range(num_updates):
            exponential_moving_average_hook.on_step(task, local_variables)
        exponential_moving_average_hook.on_phase_end(task, local_variables)
        # the model weights shouldn't have changed
        self.assertTrue(torch.allclose(model.fc.weight, fc_weight))

        # simulate a test phase now
        task.train = False
        exponential_moving_average_hook.on_phase_start(task, local_variables)
        exponential_moving_average_hook.on_phase_end(task, local_variables)

        # the model weights should be updated to the ema weights
        self.assertTrue(
            torch.allclose(
                model.fc.weight, fc_weight * (1 - math.pow(1 - decay, num_updates))
            )
        )

        # simulate a train phase again
        task.train = True
        exponential_moving_average_hook.on_phase_start(task, local_variables)

        # the model weights should be back to the old value
        self.assertTrue(torch.allclose(model.fc.weight, fc_weight))

    def test_get_model_state_iterator(self):
        device = "gpu" if torch.cuda.is_available() else "cpu"
        model = TestModel().to(device=self._map_device_string(device))
        decay = 0.5
        # test that we pick up the right parameters in the iterator
        for consider_bn_buffers in [True, False]:
            exponential_moving_average_hook = ExponentialMovingAverageModelHook(
                decay=decay, consider_bn_buffers=consider_bn_buffers, device=device
            )
            iterable = exponential_moving_average_hook.get_model_state_iterator(model)
            fc_found = False
            bn_found = False
            bn_buffer_found = False
            for _, param in iterable:
                if any(param is item for item in model.fc.parameters()):
                    fc_found = True
                if any(param is item for item in model.bn.parameters()):
                    bn_found = True
                if any(param is item for item in model.bn.buffers()):
                    bn_buffer_found = True
            self.assertTrue(fc_found)
            self.assertTrue(bn_found)
            self.assertEqual(bn_buffer_found, consider_bn_buffers)

    def test_exponential_moving_average_hook(self):
        device = "gpu" if torch.cuda.is_available() else "cpu"
        self._test_exponential_moving_average_hook(device, device)

    @unittest.skipUnless(torch.cuda.is_available(), "This test needs a gpu to run")
    def test_mixed_devices(self):
        """Tests that the hook works when the model and hook's device are different"""
        self._test_exponential_moving_average_hook("cpu", "gpu")
        self._test_exponential_moving_average_hook("gpu", "cpu")
