#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_model_configs

from classy_vision.generic.profiler import (
    compute_activations,
    compute_flops,
    count_params,
)
from classy_vision.models import build_model


class TestProfilerFunctions(unittest.TestCase):
    def test_complexity_calculation(self) -> None:
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
