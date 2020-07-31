#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.optim_test_util import TestOptimizer

import torch
from classy_vision.optim.param_scheduler import LinearParamScheduler
from classy_vision.optim.sgd import SGD


class TestSGDOptimizer(TestOptimizer, unittest.TestCase):
    def _get_config(self):
        return {
            "name": "sgd",
            "num_epochs": 90,
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "nesterov": False,
        }

    def _instance_to_test(self):
        return SGD

    # This test relies on the SGD update equations, which is why it's not in
    # the base class TestOptimizer
    def test_lr_step(self):
        opt = SGD()

        param = torch.tensor([0.0], requires_grad=True)
        opt.set_param_groups([param], lr=LinearParamScheduler(1, 2))

        param.grad = torch.tensor([1.0])

        self.assertAlmostEqual(opt.options_view.lr, 1.0)

        # lr=1, param should go from 0 to -1
        opt.step(where=0)
        self.assertAlmostEqual(opt.options_view.lr, 1.0)

        self.assertAlmostEqual(param.item(), -1.0, delta=1e-5)

        # lr=1.5, param should go from -1 to -1-1.5 = -2.5
        opt.step(where=0.5)
        self.assertAlmostEqual(param.item(), -2.5, delta=1e-5)

        # lr=1.9, param should go from -2.5 to -1.9-2.5 = -4.4
        opt.step(where=0.9)
        self.assertAlmostEqual(param.item(), -4.4, delta=1e-5)
