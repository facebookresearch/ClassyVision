#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.heads import ClassyVisionHead
from classy_vision.models.classy_vision_model import ClassyVisionModel


class TestClassyModule(unittest.TestCase):
    class DummyTestHead(ClassyVisionHead):
        def __init__(self, head_id: str, num_classes: int = 2):
            super().__init__(head_id, num_classes)
            self.layer = torch.nn.Linear(2, num_classes)

        def forward(self, x):
            return self.layer(x)

    class DummyTestModel(ClassyVisionModel):
        def __init__(self, num_classes: int = 2):
            super().__init__(num_classes)
            self.layer1 = self.build_attachable_block(
                "dummy_block", torch.nn.Linear(2, 2)
            )
            self.layer2 = self.build_attachable_block(
                "dummy_block2", torch.nn.Linear(2, num_classes)
            )

        def forward(self, x):
            out = self.layer1(x)
            return self.layer2(out)

    def test_head_execution(self):
        model = self.DummyTestModel()
        head = self.DummyTestHead("head_id")
        model.set_heads({"dummy_block2": {head.unique_id: head}}, True)
        input = torch.randn(1, 2)
        output = model(input)
        head_output = model.head_outputs["head_id"]
        self.assertTrue(torch.allclose(head(output), head_output))

    def test_duplicated_head_ids(self):
        model = self.DummyTestModel()
        head1 = self.DummyTestHead("head")
        head2 = self.DummyTestHead("head")
        heads = {
            "dummy_block": {head1.unique_id: head1},
            "dummy_block2": {head2.unique_id: head2},
        }
        with self.assertRaises(ValueError):
            model.set_heads(heads, True)

        head2.unique_id = "head_id2"
        model.set_heads(heads, True)

    def test_num_classes(self):
        model = self.DummyTestModel()
        self.assertEqual(model.num_classes, 2)

        head1 = self.DummyTestHead("head1", num_classes=3)
        heads = {"dummy_block": {head1.unique_id: head1}}
        model.set_heads(heads, freeze_trunk=False)
        self.assertEqual(model.num_classes, 3)

        head2 = self.DummyTestHead("head2", num_classes=4)
        heads["dummy_block2"] = {head2.unique_id: head2}
        model.set_heads(heads, freeze_trunk=False)
        self.assertEqual(model.num_classes, {head1.unique_id: 3, head2.unique_id: 4})
