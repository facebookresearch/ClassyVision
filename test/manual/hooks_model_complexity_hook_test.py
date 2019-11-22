#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import unittest
from test.generic.config_utils import get_test_classy_task, get_test_model_configs

from classy_vision.hooks import ModelComplexityHook
from classy_vision.models import build_model


class TestModelComplexityHook(unittest.TestCase):
    def test_model_complexity(self) -> None:
        """
        Test that the number of parameters and the FLOPs are calcuated correctly.
        """
        model_configs = get_test_model_configs()
        expected_mega_flops = [4122, 4274, 106152]
        expected_params = [25557032, 25028904, 43009448]
        local_variables = {}

        task = get_test_classy_task()
        task.prepare()

        # create a model complexity hook
        model_complexity_hook = ModelComplexityHook()

        for model_config, mega_flops, params in zip(
            model_configs, expected_mega_flops, expected_params
        ):
            model = build_model(model_config)

            task.base_model = model

            with self.assertLogs() as log_watcher:
                model_complexity_hook.on_start(task, local_variables)

            # there should be 2 log statements generated
            self.assertEqual(len(log_watcher.output), 2)

            # first statement - either the MFLOPs or a warning
            if mega_flops is not None:
                match = re.search(
                    r"FLOPs for forward pass: (?P<mega_flops>[-+]?\d*\.\d+|\d+) MFLOPs",
                    log_watcher.output[0],
                )
                self.assertIsNotNone(match)
                self.assertEqual(mega_flops, float(match.group("mega_flops")))
            else:
                self.assertIn(
                    "Model contains unsupported modules", log_watcher.output[0]
                )

            # second statement
            match = re.search(
                r"Number of parameters in model: (?P<params>[-+]?\d*\.\d+|\d+)",
                log_watcher.output[1],
            )
            self.assertIsNotNone(match)
            self.assertEqual(params, float(match.group("params")))
