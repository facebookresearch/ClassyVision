#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from classy_vision.hooks import ClassyHook


class TestHook(ClassyHook):
    on_rendezvous = ClassyHook._noop
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_sample = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, a, b):
        super().__init__()
        self.state.a = a
        self.state.b = b


class TestClassyHook(unittest.TestCase):
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
