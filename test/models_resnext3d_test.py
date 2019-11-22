#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from test.generic.utils import compare_model_state

import torch
from classy_vision.models import ClassyModel, build_model


class TestResNeXt3D(unittest.TestCase):
    def setUp(self):
        model_config_template = {
            "name": "resnext3d",
            "input_key": "video",
            "clip_crop_size": 112,
            "skip_transformation_type": "postactivated_shortcut",
            "frames_per_clip": 32,
            "input_planes": 3,
            "stem_name": "resnext3d_stem",
            "stem_planes": 64,
            "stem_temporal_kernel": 3,
            "stage_planes": 64,
            "num_groups": 1,
            "width_per_group": 16,
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "default_head",
                    "in_plane": 512,
                    "pool_size": (2, 7, 7),
                    "activation_func": "softmax",
                    "num_classes": 2,
                }
            ],
        }
        pbt = "postactivated_bottleneck_transformation"
        model_config_variants = [
            # ResNeXt3D-34
            {
                "residual_transformation_type": "basic_transformation",
                "num_blocks": [3, 4, 6, 3],
            },
            # ResNeXt3D-50
            {"residual_transformation_type": pbt, "num_blocks": [3, 4, 6, 3]},
            # ResNeXt3D-101
            {"residual_transformation_type": pbt, "num_blocks": [3, 4, 23, 3]},
        ]

        self.model_configs = []
        for variant in model_config_variants:
            model_config = copy.deepcopy(model_config_template)
            model_config.update(variant)

            block_idx = model_config["num_blocks"][-1]
            # attach the head at the last block
            model_config["heads"][0]["fork_block"] = "pathway0-stage4-block%d" % (
                block_idx - 1
            )

            self.model_configs.append(model_config)

        self.batchsize = 1

        self.forward_pass_configs = {
            "train": {
                # input shape: N x C x T x H x W
                "input": {"video": torch.rand(self.batchsize, 3, 16, 112, 112)},
                "model": {
                    "stem_maxpool": False,
                    "stage_temporal_stride": [1, 2, 2, 2],
                    "stage_spatial_stride": [1, 2, 2, 2],
                },
            },
            "test": {
                "input": {"video": torch.rand(self.batchsize, 3, 16, 256, 320)},
                "model": {
                    "stem_maxpool": True,
                    "stage_temporal_stride": [1, 2, 2, 2],
                    "stage_spatial_stride": [1, 2, 2, 2],
                },
            },
        }

    def test_build_model(self):
        for model_config in self.model_configs:
            model = build_model(model_config)
            self.assertTrue(isinstance(model, ClassyModel))
            self.assertTrue(
                type(model.output_shape) == tuple and len(model.output_shape) == 2
            )
            self.assertTrue(type(model.model_depth) == int)

    def test_forward_pass(self):
        for split, split_config in self.forward_pass_configs.items():
            for model_config in self.model_configs:
                forward_pass_model_config = copy.deepcopy(model_config)
                forward_pass_model_config.update(split_config["model"])

                num_classes = forward_pass_model_config["heads"][0]["num_classes"]

                model = build_model(forward_pass_model_config)
                model.train(split == "train")

                out = model(split_config["input"])

                self.assertEqual(out.size(), (self.batchsize, num_classes))

    def test_set_classy_state_plain(self):
        # We use the same model architecture to save and load a model state.
        # This is a plain use case of `set_classy_state` method
        for model_config in self.model_configs:
            model = build_model(model_config)
            model_state = model.get_classy_state()

            model2 = build_model(model_config)
            model2.set_classy_state(model_state)
            model2_state = model2.get_classy_state()
            compare_model_state(self, model_state, model2_state)

    def _get_model_config_weight_inflation(self):
        model_2d_config = {
            "name": "resnext3d",
            "frames_per_clip": 1,
            "input_planes": 3,
            "clip_crop_size": 224,
            "skip_transformation_type": "postactivated_shortcut",
            "residual_transformation_type": "postactivated_bottleneck_transformation",
            "num_blocks": [3, 4, 6, 3],
            "stem_name": "resnext3d_stem",
            "stem_planes": 64,
            "stem_temporal_kernel": 1,
            "stem_spatial_kernel": 7,
            "stem_maxpool": True,
            "stage_planes": 256,
            "stage_temporal_kernel_basis": [[1], [1], [1], [1]],
            "temporal_conv_1x1": [True, True, True, True],
            "stage_temporal_stride": [1, 1, 1, 1],
            "stage_spatial_stride": [1, 2, 2, 2],
            "num_groups": 1,
            "width_per_group": 64,
            "num_classes": 1000,
            "zero_init_residual_transform": True,
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "default_head",
                    "pool_size": [1, 7, 7],
                    "activation_func": "softmax",
                    "num_classes": 1000,
                    "fork_block": "pathway0-stage4-block2",
                    "in_plane": 2048,
                    "use_dropout": False,
                }
            ],
        }

        model_3d_config = {
            "name": "resnext3d",
            "frames_per_clip": 8,
            "input_planes": 3,
            "clip_crop_size": 224,
            "skip_transformation_type": "postactivated_shortcut",
            "residual_transformation_type": "postactivated_bottleneck_transformation",
            "num_blocks": [3, 4, 6, 3],
            "input_key": "video",
            "stem_name": "resnext3d_stem",
            "stem_planes": 64,
            "stem_temporal_kernel": 5,
            "stem_spatial_kernel": 7,
            "stem_maxpool": True,
            "stage_planes": 256,
            "stage_temporal_kernel_basis": [[3], [3, 1], [3, 1], [1, 3]],
            "temporal_conv_1x1": [True, True, True, True],
            "stage_temporal_stride": [1, 1, 1, 1],
            "stage_spatial_stride": [1, 2, 2, 2],
            "num_groups": 1,
            "width_per_group": 64,
            "num_classes": 1000,
            "freeze_trunk": False,
            "zero_init_residual_transform": True,
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "default_head",
                    "pool_size": [8, 7, 7],
                    "activation_func": "softmax",
                    "num_classes": 1000,
                    "fork_block": "pathway0-stage4-block2",
                    "in_plane": 2048,
                    "use_dropout": True,
                }
            ],
        }
        return model_2d_config, model_3d_config

    def test_set_classy_state_weight_inflation(self):
        # Get model state from a 2D ResNet model, inflate the 2D conv weights,
        # and use them to initialize 3D conv weights. This is an advanced use of
        # `set_classy_state` method.
        model_2d_config, model_3d_config = self._get_model_config_weight_inflation()
        model_2d = build_model(model_2d_config)
        model_2d_state = model_2d.get_classy_state()

        model_3d = build_model(model_3d_config)
        model_3d.set_classy_state(model_2d_state)
        model_3d_state = model_3d.get_classy_state()

        for name, weight_2d in model_2d_state["model"]["trunk"].items():
            weight_3d = model_3d_state["model"]["trunk"][name]
            if weight_2d.dim() == 5:
                # inflation only applies to conv weights
                self.assertEqual(weight_3d.dim(), 5)
                if weight_2d.shape[2] == 1 and weight_3d.shape[2] > 1:
                    weight_2d_inflated = (
                        weight_2d.repeat(1, 1, weight_3d.shape[2], 1, 1)
                        / weight_3d.shape[2]
                    )
                    self.assertTrue(torch.equal(weight_3d, weight_2d_inflated))

    def test_set_classy_state_weight_inflation_inconsistent_kernel_size(self):
        # Get model state from a 2D ResNet model, inflate the 2D conv weights,
        # and use them to initialize 3D conv weights.
        model_2d_config, model_3d_config = self._get_model_config_weight_inflation()
        # Modify conv kernel size in the stem layer of 2D model to 5, which is
        # inconsistent with the kernel size 7 used in 3D model.
        model_2d_config["stem_spatial_kernel"] = 5
        model_2d = build_model(model_2d_config)
        model_2d_state = model_2d.get_classy_state()
        model_3d = build_model(model_3d_config)
        with self.assertRaises(AssertionError):
            model_3d.set_classy_state(model_2d_state)
