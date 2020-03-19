#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.hooks import ClassyHook, build_hook, build_hooks, register_hook


@register_hook("test_hook")
class TestHook(ClassyHook):
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, a, b):
        super().__init__()
        self.state.a = a
        self.state.b = b

    @classmethod
    def from_config(cls, config):
        return TestHook(**config)


class TestClassyHook(unittest.TestCase):
    def test_hook_registry_and_builder(self):
        config = {"name": "test_hook", "a": 1, "b": 2}
        hook1 = build_hook(hook_config=config)
        self.assertTrue(isinstance(hook1, TestHook))
        self.assertTrue(hook1.state.a == 1)
        self.assertTrue(hook1.state.b == 2)

        hook_configs = [copy.deepcopy(config), copy.deepcopy(config)]
        hooks = build_hooks(hook_configs=hook_configs)
        for hook in hooks:
            self.assertTrue(isinstance(hook, TestHook))
            self.assertTrue(hook.state.a == 1)
            self.assertTrue(hook.state.b == 2)

    def test_state_dict(self):
        a = 0
        b = {1: 2, 3: [4]}
        test_hook = TestHook(a, b)
        state_dict = test_hook.get_classy_state()
        # create a new test_hook and set its state to the old hook's.
        test_hook = TestHook("", 0)
        test_hook.set_classy_state(state_dict)
        self.assertEqual(test_hook.state.a, a)
        self.assertEqual(test_hook.state.b, b)
