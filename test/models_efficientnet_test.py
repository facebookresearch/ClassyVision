#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.models import EfficientNet, build_model


class TestEfficientNetModel(unittest.TestCase):
    def get_model_config(self, use_model_name=False):
        model_config = {
            "name": "efficientnet",
            "model_params": {
                "width_coefficient": 1.1,
                "depth_coefficient": 1.2,
                "resolution": 260,
                "dropout_rate": 0.3,
            },
            "bn_momentum": 0.01,
            "bn_epsilon": 1e-3,
            "drop_connect_rate": 0.2,
            "num_classes": 1000,
            "width_divisor": 8,
            "min_width": None,
            "use_se": True,
        }
        if use_model_name:
            del model_config["model_params"]
            model_config["model_name"] = "B2"
        return model_config

    def test_build_model(self):
        """
        Test that the model builds using a config using either model_params or
        model_name.
        """
        for use_model_name in [True, False]:
            model = build_model(self.get_model_config(use_model_name=use_model_name))
            assert isinstance(model, EfficientNet)

    def test_build_preset_model(self):
        configs = [{"name": f"efficientnet_b{i}" for i in range(8)}]
        for config in configs:
            model = build_model(config)
            self.assertIsInstance(model, EfficientNet)

    def test_model_forward(self):
        image_shape = (3, 260, 260)
        num_images = (10,)
        input = torch.randn(num_images + image_shape)
        model = build_model(self.get_model_config())
        model(input)
