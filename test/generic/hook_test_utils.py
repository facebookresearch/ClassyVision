#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from classy_vision.hooks import build_hook


class HookTestBase(unittest.TestCase):
    def constructor_test_helper(
        self, arg_list, config, hook_type, hook_registry_name=None, invalid_configs=None
    ):
        hook1 = hook_type(*arg_list)
        self.assertTrue(isinstance(hook1, hook_type))

        hook2 = hook_type.from_config(config)
        self.assertTrue(isinstance(hook2, hook_type))

        if hook_registry_name is not None:
            config["name"] = hook_registry_name
            hook3 = build_hook(config)
            del config["name"]
            self.assertTrue(isinstance(hook3, hook_type))

        if invalid_configs is not None:
            # Verify assert logic works correctly
            for cfg in invalid_configs:
                with self.assertRaises(AssertionError):
                    hook_type.from_config(cfg)
