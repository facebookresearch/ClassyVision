#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from classy_vision.heads import ClassyHead
from classy_vision.models import (
    ClassyModel,
    ClassyModelHeadExecutorWrapper,
    ClassyModelWrapper,
)


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
            self.dummy_block = torch.nn.Linear(2, 2)
            self.dummy_block2 = torch.nn.Linear(2, 2)

        def forward(self, x):
            out = self.dummy_block(x)
            return self.dummy_block2(out)

    class DummyTestModelDuplicatedBlockNames(ClassyModel):
        def __init__(self):
            super().__init__()
            self.dummy_block = torch.nn.Linear(2, 2)
            self.features = nn.Sequential()
            self.features.add_module("dummy_model", torch.nn.Linear(2, 2))

        def forward(self, x):
            out = self.dummy_block(x)
            return self.features.dummy_block(out)

    def test_head_execution(self):
        orig_wrapper_cls = self.DummyTestModel.wrapper_cls

        # test head outputs without any extra wrapper logic, which is the case with
        # no wrappers or the base ClassyModelWrapper class
        for wrapper_class in [None, ClassyModelWrapper]:
            self.DummyTestModel.wrapper_cls = wrapper_class
            model = self.DummyTestModel()
            head = self.DummyTestHead()
            model.set_heads({"dummy_block2": [head]})
            input = torch.randn(1, 2)
            output = model(input)
            head_output = model.execute_heads()
            self.assertTrue(torch.allclose(head(output), head_output["head_id"]))

        # test that the head output is returned automatically with the
        # ClassyModelHeadExecutorWrapper
        self.DummyTestModel.wrapper_cls = ClassyModelHeadExecutorWrapper
        model = self.DummyTestModel()
        head = self.DummyTestHead()
        model.set_heads({"dummy_block2": [head]})
        input = torch.randn(1, 2)
        output = model(input)
        head_output = model.execute_heads()
        self.assertTrue(torch.allclose(output, head_output["head_id"]))

        # restore the wrapper class
        self.DummyTestModel.wrapper_cls = orig_wrapper_cls

    def test_duplicated_head_ids(self):
        model = self.DummyTestModel()
        head1 = self.DummyTestHead()
        head2 = self.DummyTestHead()
        heads = {"dummy_block": [head1], "dummy_block2": [head2]}
        with self.assertRaises(ValueError):
            model.set_heads(heads)

        head2.unique_id = "head_id2"
        model.set_heads(heads)

    def test_duplicated_block_names(self):
        model = self.DummyTestModelDuplicatedBlockNames()
        head = self.DummyTestHead()
        heads = {"dummy_block2": [head]}
        with self.assertRaises(Exception):
            # there are two modules with the name "dummy_block2"
            # which is not supported
            model.set_heads(heads)
        # can still attach to a module with a unique id
        heads = {"features": [head]}
        model.set_heads(heads)

    def test_set_heads(self):
        model = self.DummyTestModel()
        head = self.DummyTestHead()
        self.assertEqual(
            len(model.get_heads()), 0, "heads should be empty before set_heads"
        )
        model.set_heads({"dummy_block2": [head]})
        input = torch.randn(1, 2)
        model(input)
        head_outputs = model.execute_heads()
        self.assertEqual(len(head_outputs), 1, "should have output for one head")

        # remove all heads
        model.set_heads({})
        self.assertEqual(len(model.get_heads()), 0, "heads should be empty")

        # try a non-existing module
        with self.assertRaises(Exception):
            model.set_heads({"unknown_block": [head]})
