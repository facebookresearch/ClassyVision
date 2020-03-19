#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_classy_task, get_test_model_configs

from classy_vision.hooks import ModelComplexityHook, build_hook
from classy_vision.models import build_model


class TestModelComplexityHook(unittest.TestCase):
    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        config = {}

        hook1 = ModelComplexityHook()
        hook2 = ModelComplexityHook.from_config(config)
        config["name"] = "model_complexity"
        hook3 = build_hook(config)
        del config["name"]

        self.assertTrue(isinstance(hook1, ModelComplexityHook))
        self.assertTrue(isinstance(hook2, ModelComplexityHook))
        self.assertTrue(isinstance(hook3, ModelComplexityHook))

    def test_model_complexity_hook(self) -> None:
        model_configs = get_test_model_configs()

        task = get_test_classy_task()
        task.prepare()

        # create a model complexity hook
        model_complexity_hook = ModelComplexityHook()

        for model_config in model_configs:
            model = build_model(model_config)

            task.base_model = model

            with self.assertLogs():
                model_complexity_hook.on_start(task)
