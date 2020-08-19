#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import classy_vision.heads.fully_convolutional_linear_head as fcl
import torch


class TestFullyConvolutionalLinearHead(unittest.TestCase):
    def test_fully_convolutional_linear_head(self):
        head = fcl.FullyConvolutionalLinearHead(
            "default_head",
            num_classes=2,
            in_plane=3,
            pool_size=[1, 3, 3],
            activation_func="softmax",
            use_dropout=False,
        )
        input = torch.rand([1, 3, 4, 3, 3])
        output = head(input)
        self.assertEqual(output.shape, torch.Size([1, 8]))

    def test_fully_convolutional_linear_head_eval(self):
        head = fcl.FullyConvolutionalLinearHead(
            "default_head",
            num_classes=2,
            in_plane=3,
            pool_size=[1, 3, 3],
            activation_func="softmax",
            use_dropout=False,
        ).eval()
        input = torch.rand([1, 3, 4, 3, 3])
        output = head(input)
        self.assertEqual(output.shape, torch.Size([1, 2]))

    def test_fully_convolutional_linear_head_from_cfg(self):
        head_cfg = {
            "name": "fully_convolutional_linear",
            "unique_id": "default_head",
            "activation_func": "softmax",
            "pool_size": [1, 3, 3],
            "num_classes": 2,
            "in_plane": 3,
            "use_dropout": False,
        }
        head = fcl.FullyConvolutionalLinearHead.from_config(head_cfg)
        input = torch.rand([1, 3, 4, 3, 3])
        output = head(input)
        self.assertEqual(output.shape, torch.Size([1, 8]))

    def test_fully_convolutional_linear_head_adaptive_pool(self):
        head = fcl.FullyConvolutionalLinearHead(
            "default_head",
            num_classes=2,
            in_plane=3,
            pool_size=None,
            activation_func="softmax",
            use_dropout=False,
        )
        input = torch.rand([1, 3, 4, 3, 3])
        output = head(input)
        self.assertEqual(output.shape, torch.Size([1, 2]))

    def test_fully_convolutional_linear_head_adaptive_pool_from_cfg(self):
        head_cfg = {
            "name": "fully_convolutional_linear",
            "unique_id": "default_head",
            "activation_func": "softmax",
            "num_classes": 2,
            "in_plane": 3,
            "use_dropout": False,
        }
        head = fcl.FullyConvolutionalLinearHead.from_config(head_cfg)
        input = torch.rand([1, 3, 4, 3, 3])
        output = head(input)
        self.assertEqual(output.shape, torch.Size([1, 2]))
