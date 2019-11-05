#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.heads import ClassyHead, build_head, register_head


class TestClassyHead(unittest.TestCase):
    @register_head("dummy_head")
    class DummyHead(ClassyHead):
        def __init__(self, unique_id, num_classes, in_plane):
            super().__init__(unique_id, num_classes)
            self.fc = torch.nn.Linear(in_plane, num_classes)

        def forward(self, x):
            return self.fc(x)

        @classmethod
        def from_config(cls, config):
            return cls(config["unique_id"], config["num_classes"], config["in_plane"])

    def _get_config(self):
        return {
            "name": "dummy_head",
            "num_classes": 3,
            "unique_id": "cortex_dummy_head",
            "fork_block": "block3",
            "in_plane": 2048,
        }

    def test_build_head(self):
        config = self._get_config()
        head = build_head(config)
        self.assertEqual(head.unique_id, config["unique_id"])

        del config["unique_id"]
        with self.assertRaises(AssertionError):
            head = build_head(config)

    def test_forward(self):
        config = self._get_config()
        head = build_head(config)
        input = torch.randn(1, config["in_plane"])
        output = head(input)
        self.assertEqual(output.size(), torch.Size([1, 3]))

    def _get_pass_through_config(self):
        return {
            "name": "identity",
            "num_classes": 3,
            "unique_id": "cortex_pass_through_head",
            "fork_block": "block3",
            "in_plane": 4,
        }

    def test_identity_forward(self):
        config = self._get_pass_through_config()
        head = build_head(config)
        input = torch.randn(1, config["in_plane"])
        output = head(input)
        self.assertEqual(input.size(), output.size())
        self.assert_(torch.all(torch.eq(input, output)))
