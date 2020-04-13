#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_classy_task, get_test_model_configs
from test.generic.hook_test_utils import HookTestBase

from classy_vision.hooks import ModelComplexityHook
from classy_vision.models import build_model


class TestModelComplexityHook(HookTestBase):
    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        config = {}

        self.constructor_test_helper(
            config=config,
            hook_type=ModelComplexityHook,
            hook_registry_name="model_complexity",
        )

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
