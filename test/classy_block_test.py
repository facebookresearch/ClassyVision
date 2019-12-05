#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.heads import ClassyHead
from classy_vision.models import ClassyModel


class TestClassyBlock(unittest.TestCase):
    class DummyTestHead(ClassyHead):
        def __init__(self):
            super().__init__("head_id")
            self.layer = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.layer(x)

    class DummyTestModel(ClassyModel):
        def __init__(self):
            super().__init__()
            self.layer1 = self.build_attachable_block(
                "dummy_block", torch.nn.Linear(2, 2)
            )
            self.layer2 = self.build_attachable_block(
                "dummy_block2", torch.nn.Linear(2, 2)
            )

        def forward(self, x):
            out = self.layer1(x)
            return self.layer2(out)

    def test_head_execution(self):
        model = self.DummyTestModel()
        head = self.DummyTestHead()
        model.set_heads({"dummy_block2": {head.unique_id: head}})
        input = torch.randn(1, 2)
        output = model(input)
        head_output = model.execute_heads()
        self.assertTrue(torch.allclose(head(output), head_output["head_id"]))

    def test_duplicated_head_ids(self):
        model = self.DummyTestModel()
        head1 = self.DummyTestHead()
        head2 = self.DummyTestHead()
        heads = {
            "dummy_block": {head1.unique_id: head1},
            "dummy_block2": {head2.unique_id: head2},
        }
        with self.assertRaises(ValueError):
            model.set_heads(heads)

        head2.unique_id = "head_id2"
        model.set_heads(heads)

    def test_set_heads(self):
        model = self.DummyTestModel()
        head = self.DummyTestHead()
        self.assertEqual(
            len(model.get_heads()), 0, "heads should be empty before set_heads"
        )
        model.set_heads({"dummy_block2": {head.unique_id: head}})
        input = torch.randn(1, 2)
        model(input)
        head_outputs = model.execute_heads()
        self.assertEqual(len(head_outputs), 1, "should have output for one head")

        # remove all heads
        model.set_heads({})
        self.assertEqual(len(model.get_heads()), 0, "heads should be empty")
