#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_model_configs

import torch
import torch.nn as nn
from classy_vision.generic.profiler import (
    compute_activations,
    compute_flops,
    count_params,
)
from classy_vision.models import build_model


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        # add parameters to the module to affect the parameter count
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x):
        return x + 1

    def flops(self, x):
        # TODO: this should raise an exception if this function is not defined
        # since the FLOPs are indeterminable

        # need to define flops since this is an unknown class
        return x.numel()


class TestConvModule(nn.Conv2d):
    def __init__(self):
        super().__init__(2, 3, (4, 4), bias=False)
        # add another (unused) layer for added complexity and to test parameters
        self.linear = nn.Linear(4, 5, bias=False)

    def forward(self, x):
        return x

    def activations(self, x, out):
        # TODO: this should ideally work without this function being defined
        return out.numel()

    def flops(self, x):
        # need to define flops since this is an unknown class
        return 0


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(300, 300, bias=False)
        self.mod = TestModule()
        self.conv = TestConvModule()
        # we should be able to pick up user defined parameters as well
        self.extra_params = nn.Parameter(torch.randn(10, 10))
        # we shouldn't count flops for an unused layer
        self.unused_linear = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.mod(out)
        return self.linear(out)


class TestProfilerFunctions(unittest.TestCase):
    def test_complexity_calculation_resnext(self) -> None:
        model_configs = get_test_model_configs()
        # make sure there are three configs returned
        self.assertEqual(len(model_configs), 3)

        # expected values which allow minor deviations from model changes
        # we only test at the 10^6 scale
        expected_m_flops = [4122, 7850, 8034]
        expected_m_params = [25, 44, 44]
        expected_m_activations = [11, 16, 21]

        for model_config, m_flops, m_params, m_activations in zip(
            model_configs, expected_m_flops, expected_m_params, expected_m_activations
        ):
            model = build_model(model_config)
            self.assertEqual(compute_activations(model) // 10 ** 6, m_activations)
            self.assertEqual(compute_flops(model) // 10 ** 6, m_flops)
            self.assertEqual(count_params(model) // 10 ** 6, m_params)

    def test_complexity_calculation(self) -> None:
        model = TestModel()
        input_shape = (3, 10, 10)
        num_elems = 3 * 10 * 10
        self.assertEqual(compute_activations(model, input_shape=input_shape), num_elems)
        self.assertEqual(
            compute_flops(model, input_shape=input_shape),
            num_elems
            + 0
            + (300 * 300),  # TestModule + TestConvModule + TestModel.linear;
            # TestModel.unused_linear is unused and shouldn't be counted
        )
        self.assertEqual(
            count_params(model),
            (2 * 3) + (2 * 3 * 4 * 4) + (4 * 5) + (300 * 300) + (10 * 10) + (2 * 2),
        )  # TestModule.linear + TestConvModule + TestConvModule.linear +
        # TestModel.linear + TestModel.extra_params + TestModel.unused_linear
