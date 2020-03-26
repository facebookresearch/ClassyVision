#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import unittest
from test.generic.utils import compare_model_state

import torch
from classy_vision.models import ResNeXt, build_model


MODELS = {
    "small_resnext": {
        "name": "resnext",
        "num_blocks": [1, 1, 1, 1],
        "init_planes": 4,
        "reduction": 4,
        "base_width_and_cardinality": [2, 32],
        "small_input": True,
        "zero_init_bn_residuals": True,
        "basic_layer": True,
        "final_bn_relu": True,
        "heads": [
            {
                "name": "fully_connected",
                "unique_id": "default_head",
                "num_classes": 1000,
                "fork_block": "block3-0",
                "in_plane": 128,
            }
        ],
    },
    "small_resnet": {
        "name": "resnet",
        "num_blocks": [1, 1, 1, 1],
        "init_planes": 4,
        "reduction": 4,
        "small_input": True,
        "zero_init_bn_residuals": True,
        "basic_layer": True,
        "final_bn_relu": True,
        "heads": [
            {
                "name": "fully_connected",
                "unique_id": "default_head",
                "num_classes": 1000,
                "fork_block": "block3-0",
                "in_plane": 128,
            }
        ],
    },
    "small_resnet_se": {
        "name": "resnet",
        "num_blocks": [1, 1, 1, 1],
        "init_planes": 4,
        "reduction": 4,
        "small_input": True,
        "zero_init_bn_residuals": True,
        "basic_layer": True,
        "final_bn_relu": True,
        "use_se": True,
        "heads": [
            {
                "name": "fully_connected",
                "unique_id": "default_head",
                "num_classes": 1000,
                "fork_block": "block3-0",
                "in_plane": 128,
            }
        ],
    },
}


class TestResnext(unittest.TestCase):
    def _test_model(self, model_config):
        """This test will build ResNeXt-* models, run a forward pass and
        verify output shape, and then verify that get / set state
        works.

        I do this in one test so that we construct the model a minimum
        number of times.
        """
        model = build_model(model_config)

        model.eval()  # eval mode to make sure batchnorm stats don't change

        # Verify forward pass works
        input = torch.ones([1, 3, 32, 32])
        ser_before = pickle.dumps(model)
        output = model.forward(input)
        ser_after = pickle.dumps(model)

        self.assertEqual(output.size(), (1, 1000))
        # Ensure that execution of the forward pass is stateless - i.e. we don't
        # set any attributes
        self.assertEqual(ser_before, ser_after)

        # Verify get_set_state
        new_model = build_model(model_config)
        state = model.get_classy_state()
        new_model.set_classy_state(state)
        new_state = new_model.get_classy_state()

        compare_model_state(self, state, new_state, check_heads=True)

    def test_build_preset_model(self):
        configs = [
            {"name": "resnet18"},
            {"name": "resnet18", "use_se": True},
            {
                "name": "resnet50",
                "heads": [
                    {
                        "name": "fully_connected",
                        "unique_id": "default_head",
                        "num_classes": 1000,
                        "fork_block": "block3-2",
                        "in_plane": 2048,
                    }
                ],
            },
            {
                "name": "resnext50_32x4d",
                "heads": [
                    {
                        "name": "fully_connected",
                        "unique_id": "default_head",
                        "num_classes": 1000,
                        "fork_block": "block3-2",
                        "in_plane": 2048,
                    }
                ],
            },
        ]
        for config in configs:
            model = build_model(config)
            self.assertIsInstance(model, ResNeXt)

    def test_small_resnext(self):
        self._test_model(MODELS["small_resnext"])

    def test_small_resnet(self):
        self._test_model(MODELS["small_resnet"])

    def test_small_resnet_se(self):
        self._test_model(MODELS["small_resnet_se"])
