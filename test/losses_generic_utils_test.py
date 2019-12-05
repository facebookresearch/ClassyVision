#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.generic.util import convert_to_one_hot


class TestUtils(unittest.TestCase):
    def test_single(self):
        targets = torch.tensor([[4]])
        one_hot_target = convert_to_one_hot(targets, 5)
        self.assertTrue(torch.allclose(one_hot_target, torch.tensor([[0, 0, 0, 0, 1]])))

    def test_two(self):
        targets = torch.tensor([[0], [1]])
        one_hot_target = convert_to_one_hot(targets, 3)
        self.assertTrue(
            torch.allclose(one_hot_target, torch.tensor([[1, 0, 0], [0, 1, 0]]))
        )
