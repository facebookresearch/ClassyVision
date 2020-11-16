#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.utils import compare_model_state
import copy
import torch
from classy_vision.models import build_model


class TestVisionTransformer(unittest.TestCase):
    def get_vit_b_16_224_config(self):
        return {
            "name": "vision_transformer",
            "image_size": 224,
            "patch_size": 16,
            "hidden_dim": 768,
            "mlp_dim": 3072,
            "num_heads": 12,
            "num_layers": 12,
            "attention_dropout_rate": 0,
            "dropout_rate": 0.1,
            "heads": [
                {
                    "name": "vision_transformer_head",
                    "unique_id": "default_head",
                    "num_classes": 1000,
                    "fork_block": "trunk_output",
                    "in_plane": 768,
                    "hidden_dim": 3072,
                }
            ],
        }

    def get_vit_l_32_224_config(self):
        return {
            "name": "vision_transformer",
            "image_size": 224,
            "patch_size": 32,
            "hidden_dim": 1024,
            "mlp_dim": 4096,
            "num_heads": 16,
            "num_layers": 24,
            "attention_dropout_rate": 0,
            "dropout_rate": 0.1,
            "heads": [
                {
                    "name": "vision_transformer_head",
                    "unique_id": "default_head",
                    "num_classes": 1000,
                    "fork_block": "trunk_output",
                    "in_plane": 1024,
                    "hidden_dim": 4096,
                }
            ],
        }

    def _test_model(self, model_config, image_size=224, expected_out_dims=1000):
        model = build_model(model_config)

        # Verify forward pass works
        input = torch.ones([2, 3, image_size, image_size])
        output = model.forward(input)
        self.assertEqual(output.size(), (2, expected_out_dims))

        # Verify get_set_state
        new_model = build_model(model_config)
        state = model.get_classy_state()
        new_model.set_classy_state(state)
        new_state = new_model.get_classy_state()

        compare_model_state(self, state, new_state, check_heads=True)

    def test_vit_b_16_224(self):
        self._test_model(self.get_vit_b_16_224_config())

    def test_vit_l_32_224(self):
        self._test_model(self.get_vit_l_32_224_config())

    def test_all_presets(self):
        for model_name, image_size, expected_out_dims in [
            ("vit_b_32", 32, 768),
            ("vit_b_16", 64, 768),
            ("vit_l_32", 32, 1024),
            ("vit_l_16", 32, 1024),
            ("vit_h_14", 14, 1280),
        ]:
            self._test_model(
                {"name": model_name, "image_size": image_size},
                image_size,
                expected_out_dims,
            )

    def test_resolution_change(self):
        vit_b_16_224_config = self.get_vit_b_16_224_config()
        vit_b_16_896_config = copy.deepcopy(vit_b_16_224_config)
        vit_b_16_896_config["image_size"] = 896

        vit_b_16_224_model = build_model(vit_b_16_224_config)
        vit_b_16_896_model = build_model(vit_b_16_896_config)

        # test state transfer from both resolutions
        vit_b_16_224_model.set_classy_state(vit_b_16_896_model.get_classy_state())
        vit_b_16_896_model.set_classy_state(vit_b_16_224_model.get_classy_state())

        vit_b_16_448_config = copy.deepcopy(vit_b_16_224_config)
        vit_b_16_448_config["image_size"] = 448
        vit_b_16_448_model = build_model(vit_b_16_448_config)

        # downsampling from 896 -> 448 -> 224 should give similar results to 896 -> 224
        vit_b_16_448_model.set_classy_state(vit_b_16_896_model.get_classy_state())
        vit_b_16_224_model.set_classy_state(vit_b_16_448_model.get_classy_state())

        vit_b_16_224_model_2 = build_model(vit_b_16_224_config)
        vit_b_16_224_model_2.set_classy_state(vit_b_16_896_model.get_classy_state())

        # we should have similar position embeddings in both models
        state_1 = vit_b_16_224_model.get_classy_state()["model"]["trunk"][
            "encoder.pos_embedding"
        ]
        state_2 = vit_b_16_224_model_2.get_classy_state()["model"]["trunk"][
            "encoder.pos_embedding"
        ]
        diff = state_1 - state_2
        self.assertLess(diff.norm() / min(state_1.norm(), state_2.norm()), 0.1)
