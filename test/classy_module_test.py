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
        def __init__(self):
            super().__init__({"name": "dummy_head", "unique_id": "head_id"})
            self.layer = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.layer(x)

    class DummyTestModel(ClassyVisionModel):
        def __init__(self):
            super().__init__({})
            self.layer1 = self.build_attachable_block(
                "dummy_block", torch.nn.Linear(2, 2)
            )
            self.layer2 = self.build_attachable_block(
                "dummy_block2", torch.nn.Linear(2, 2)
            )

        def forward(self, x):
            out = self.layer1(x)
            return self.layer2(out)

    class DummyTestModelWithDictInput(DummyTestModel):
        def requires_dict_input(self):
            return True

    class DummyTestHeadWithDictInput(DummyTestHead):
        def forward(self, x):
            return self.layer(x["data"]) + x["extra_feature"]

        def requires_dict_input(self):
            return True

    def test_head_execution(self):
        model = self.DummyTestModel()
        head = self.DummyTestHead()
        model.set_heads({"dummy_block2": {head.unique_id: head}}, True)
        input = torch.randn(1, 2)
        output = model(input)
        head_output = model.head_outputs["head_id"]
        self.assertTrue(torch.allclose(head(output), head_output))

    def test_duplicated_head_ids(self):
        model = self.DummyTestModel()
        head1 = self.DummyTestHead()
        head2 = self.DummyTestHead()
        heads = {
            "dummy_block": {head1.unique_id: head1},
            "dummy_block2": {head2.unique_id: head2},
        }
        with self.assertRaises(ValueError):
            model.set_heads(heads, True)

        head2._config["unique_id"] = "head_id2"
        model.set_heads(heads, True)

    def test_head_with_dict_input(self):
        model = self.DummyTestModelWithDictInput()
        head = self.DummyTestHeadWithDictInput()
        heads = {"dummy_block": {head.unique_id: head}}
        model.set_heads(heads, True)
        input = {"data": torch.rand(1, 2), "extra_feature": torch.rand(1, 2)}
        with torch.no_grad():
            model(input)
