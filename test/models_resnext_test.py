#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import unittest
from test.generic.utils import compare_model_state

import torch
import torchvision.models
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

        # Verify forward pass works
        input = torch.ones([1, 3, 32, 32])
        output = model.forward(input)
        self.assertEqual(output.size(), (1, 1000))

        # Verify get_set_state
        new_model = build_model(model_config)
        state = model.get_classy_state()
        new_model.set_classy_state(state)
        new_state = new_model.get_classy_state()

        compare_model_state(self, state, new_state, check_heads=True)

    def test_build_preset_model(self):
        configs = [
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


class TestTorchvisionEquivalence(unittest.TestCase):
    @staticmethod
    def tensor_sizes(state):
        size_count = collections.defaultdict(int)
        for key, value in state.items():
            if key.startswith("fc."):
                continue  # "head" for torchvision
            size_count[value.size()] += 1
        return dict(size_count)

    def assert_tensor_sizes_match_torchvision(self, model_name):
        classy_model = build_model({"name": model_name})
        torchvision_model = getattr(torchvision.models, model_name)(pretrained=False)
        classy_sizes = self.tensor_sizes(
            classy_model.get_classy_state()["model"]["trunk"]
        )
        torchvision_sizes = self.tensor_sizes(torchvision_model.state_dict())
        self.assertEqual(
            classy_sizes,
            torchvision_sizes,
            f"{model_name} tensor shapes do not match torchvision",
        )

    def test_resnet18(self):
        """Resnet18 tensor shapes should match torchvision."""
        self.assert_tensor_sizes_match_torchvision("resnet18")

    def test_resnet34(self):
        """Resnet34 tensor shapes should match torchvision."""
        self.assert_tensor_sizes_match_torchvision("resnet34")

    def test_resnet50(self):
        """Resnet50 tensor shapes should match torchvision."""
        self.assert_tensor_sizes_match_torchvision("resnet50")

    def test_resnext50_32x4d(self):
        """Resnext50_32x4d tensor shapes should match torchvision."""
        self.assert_tensor_sizes_match_torchvision("resnext50_32x4d")
