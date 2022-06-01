#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from functools import partial

import torch
from classy_vision.generic.util import get_torch_version
from classy_vision.heads import build_head
from classy_vision.heads.fully_connected_head import FullyConnectedHead
from test.generic.utils import ClassyTestCase


class TestFullyConnectedHead(ClassyTestCase):
    def test_fully_connected_head(self):
        batch_size = 2
        in_plane = 3
        image_size = 4
        num_classes = 5
        head = FullyConnectedHead(
            "default_head",
            num_classes=num_classes,
            in_plane=in_plane,
        )
        input = torch.rand([batch_size, in_plane, image_size, image_size])
        output = head(input)
        self.assertEqual(output.shape, torch.Size([batch_size, num_classes]))

    def test_fully_connected_head_normalize_inputs(self):
        batch_size = 2
        in_plane = 3
        image_size = 4
        head = FullyConnectedHead(
            "default_head",
            in_plane=in_plane,
            normalize_inputs="l2",
            num_classes=None,
        )
        input = torch.rand([batch_size, in_plane, image_size, image_size])
        output = head(input)
        self.assertEqual(output.shape, torch.Size([batch_size, in_plane]))
        for i in range(batch_size):
            output_i = output[i]
            self.assertAlmostEqual(output_i.norm().item(), 1, delta=1e-5)

        # test that the grads will be the same when using normalization as when
        # normalizing an input and passing it to the head without normalize_inputs.
        # use input with a norm > 1 and make image_size = 1 so that average
        # pooling is a no op
        image_size = 1
        input = 2 + torch.rand([batch_size, in_plane, image_size, image_size])
        norm_func = (
            torch.linalg.norm
            if get_torch_version() >= [1, 7]
            else partial(torch.norm, p=2)
        )
        norms = norm_func(input.view(batch_size, -1), dim=1)
        normalized_input = torch.clone(input)
        for i in range(batch_size):
            normalized_input[i] /= norms[i]
        num_classes = 10
        head_norm = FullyConnectedHead(
            "default_head",
            in_plane=in_plane,
            normalize_inputs="l2",
            num_classes=num_classes,
        )
        head_no_norm = FullyConnectedHead(
            "default_head",
            in_plane=in_plane,
            num_classes=num_classes,
        )
        # share the weights between the heads
        head_norm.load_state_dict(copy.deepcopy(head_no_norm.state_dict()))

        # use the sum of the output as the loss and perform a backward
        head_no_norm(normalized_input).sum().backward()
        head_norm(input).sum().backward()

        for param_1, param_2 in zip(head_norm.parameters(), head_no_norm.parameters()):
            self.assertTorchAllClose(param_1, param_2)
            self.assertTorchAllClose(param_1.grad, param_2.grad)

    def test_conv_planes(self):
        num_classes = 10
        in_plane = 3
        conv_planes = 5
        batch_size = 2
        image_size = 4
        head_config = {
            "name": "fully_connected",
            "unique_id": "asd",
            "in_plane": in_plane,
            "conv_planes": conv_planes,
            "num_classes": num_classes,
        }
        head = build_head(head_config)
        self.assertIsInstance(head, FullyConnectedHead)

        # specify an activation
        head_config["activation"] = "relu"
        head = build_head(head_config)

        # make sure that the head runs and returns the expected dimensions
        input = torch.rand([batch_size, in_plane, image_size, image_size])
        output = head(input)
        self.assertEqual(output.shape, (batch_size, num_classes))
