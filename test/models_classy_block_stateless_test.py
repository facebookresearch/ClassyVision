#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
from classy_vision.models import ClassyBlock


class TestClassyStatelessBlock(unittest.TestCase):
    def setUp(self):
        """
        This test checks on output stateful (default) and stateless variants of ClassyBlock
        by enabling and propagating the environmental variable CLASSY_BLOCK_STATELESS
        """
        # initialize stateful model
        self._model_stateful = ClassyBlock(name="stateful", module=torch.nn.Identity())
        # initialize stateless model
        os.environ["CLASSY_BLOCK_STATELESS"] = "1"
        self._model_stateless = ClassyBlock(
            name="stateless", module=torch.nn.Identity()
        )
        # note: use low=1 since default of ClassyBlock output variable is torch.zeros
        self._data = torch.randint(low=1, high=5, size=(3, 5, 5))

    def tearDown(self):
        # environmental variables do not propagate outside the scope of this test
        # but we'll clean it up anyway
        del os.environ["CLASSY_BLOCK_STATELESS"]

    def test_classy_output_stateless(self):
        # confirm model.output is (stateless) i.e. default of torch.zeros(0) and
        # that output == data
        output = self._model_stateless.forward(self._data)
        self.assertTrue(torch.equal(self._model_stateless.output, torch.zeros(0)))
        self.assertTrue(torch.equal(output, self._data))

    def test_classy_output_stateful(self):
        # confirm model.output keeps input data and that output == data
        output = self._model_stateful.forward(self._data)
        self.assertTrue(torch.equal(self._model_stateful.output, output))
        self.assertTrue(torch.equal(output, self._data))
