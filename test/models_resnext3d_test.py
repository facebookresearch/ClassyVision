#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from classy_vision.models import ClassyModel, build_model


class TestResNeXt3D(unittest.TestCase):
    def setUp(self):
        model_config_template = {
            "name": "resnext3d",
            "input_key": "video",
            "clip_crop_size": 112,
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
        model_config_variants = [
            # ResNeXt3D-34
            {"transformation_type": "basic_transformation", "num_blocks": [3, 4, 6, 3]},
            # ResNeXt3D-50
            {
                "transformation_type": "bottleneck_transformation",
                "num_blocks": [3, 4, 6, 3],
            },
            # ResNeXt3D-101
            {
                "transformation_type": "bottleneck_transformation",
                "num_blocks": [3, 4, 23, 3],
            },
        ]

        self.model_configs = []
        for variant in model_config_variants:
            model_config = copy.deepcopy(model_config_template)
            model_config.update(variant)

            block_idx = model_config["num_blocks"][-1]
            # attach the head at the last block
            model_config["heads"][0]["fork_block"] = (
                "pathway1-stage5-block%d" % block_idx
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
