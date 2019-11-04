#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.models import ClassyModel, build_model


class TestMLPModel(unittest.TestCase):
    def test_build_model(self):
        config = {"name": "mlp", "input_dim": 3, "output_dim": 1, "hidden_dims": [2]}
        model = build_model(config)
        self.assertTrue(isinstance(model, ClassyModel))
        self.assertEqual(model.model_depth, 2)

        tensor = torch.tensor([[1, 2, 3]], dtype=torch.float)
        output = model.forward(tensor)
        self.assertEqual(output.shape, torch.Size([1, 1]))

        tensor = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float)
        output = model.forward(tensor)
        self.assertEqual(output.shape, torch.Size([2, 1]))
