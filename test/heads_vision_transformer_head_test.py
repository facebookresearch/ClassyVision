#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.heads.vision_transformer_head import VisionTransformerHead


class TestVisionTransformerHead(unittest.TestCase):
    def test_vision_transformer_head(self):
        batch_size = 2
        in_plane = 3
        num_classes = 5
        head = VisionTransformerHead(
            "default_head",
            num_classes=num_classes,
            in_plane=in_plane,
        )
        input = torch.rand([batch_size, in_plane])
        output = head(input)
        self.assertEqual(output.shape, torch.Size([batch_size, num_classes]))

    def test_vision_transformer_head_normalize_inputs(self):
        batch_size = 2
        in_plane = 3
        head = VisionTransformerHead(
            "default_head",
            num_classes=None,
            in_plane=in_plane,
            normalize_inputs="l2",
        )
        input = torch.rand([batch_size, in_plane])
        output = head(input)
        self.assertEqual(output.shape, torch.Size([batch_size, in_plane]))
        for i in range(batch_size):
            output_i = output[i]
            self.assertAlmostEqual(output_i.norm().item(), 1, places=3)
