#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.generic.util import get_torch_version
from classy_vision.models import build_model, ClassyModel


class TestMLPModel(unittest.TestCase):
    def test_build_model(self):
        config = {"name": "mlp", "input_dim": 3, "output_dim": 1, "hidden_dims": [2]}
        model = build_model(config)
        self.assertTrue(isinstance(model, ClassyModel))

        tensor = torch.tensor([[1, 2, 3]], dtype=torch.float)
        output = model.forward(tensor)
        self.assertEqual(output.shape, torch.Size([1, 1]))

        tensor = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float)
        output = model.forward(tensor)
        self.assertEqual(output.shape, torch.Size([2, 1]))

    @unittest.skipIf(
        get_torch_version() < [1, 13],
        "This test is using a new api of FX Graph Mode Quantization which is only available after 1.13",
    )
    def test_quantize_model(self):
        import torch.ao.quantization as tq
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        config = {"name": "mlp", "input_dim": 3, "output_dim": 1, "hidden_dims": [2]}
        model = build_model(config)
        self.assertTrue(isinstance(model, ClassyModel))

        model.eval()
        example_inputs = (torch.rand(1, 3),)
        model.mlp = prepare_fx(model.mlp, {"": tq.default_qconfig}, example_inputs)
        model.mlp = convert_fx(model.mlp)

        tensor = torch.tensor([[1, 2, 3]], dtype=torch.float)
        output = model.forward(tensor)
        self.assertEqual(output.shape, torch.Size([1, 1]))

        tensor = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float)
        output = model.forward(tensor)
        self.assertEqual(output.shape, torch.Size([2, 1]))
