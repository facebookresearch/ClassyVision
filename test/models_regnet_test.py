#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
from classy_vision.models import RegNet, build_model
from parameterized import parameterized


# Test the different exposed parameters, even if not present in the
# actual checked in configurations
REGNET_TEST_CONFIGS = [
    (
        {
            # RegNetY
            "name": "regnet",
            "bn_epsilon": 1e-05,  # optional
            "bn_momentum": 0.1,  # optional
            "stem_type": "simple_stem_in",  # optional
            "stem_width": 32,  # optional
            "block_type": "res_bottleneck_block",  # optional
            "depth": 22,
            "w_0": 24,
            "w_a": 24.48,
            "w_m": 2.54,
            "group_width": 16,
            "use_se": True,  # optional
            "se_ratio": 0.25,  # optional
        },
    ),
    (
        {
            # RegNetX-like (no se)
            "name": "regnet",
            "depth": 22,
            "w_0": 24,
            "w_a": 24.48,
            "w_m": 2.54,
            "group_width": 16,
            "use_se": False,  # optional
        },
    ),
    (
        {
            # RegNetY, different block
            "name": "regnet",
            "depth": 22,
            "w_0": 24,
            "w_a": 24.48,
            "w_m": 2.54,
            "group_width": 16,
            "block_type": "vanilla_block",  # optional
        },
    ),
    (
        {
            # RegNetY, different block
            "name": "regnet",
            "depth": 22,
            "w_0": 24,
            "w_a": 24.48,
            "w_m": 2.54,
            "group_width": 16,
            "block_type": "res_basic_block",  # optional
        },
    ),
    (
        {
            # RegNetY, different stem
            "name": "regnet",
            "depth": 22,
            "w_0": 24,
            "w_a": 24.48,
            "w_m": 2.54,
            "group_width": 16,
            "stem_type": "res_stem_cifar",  # optional
        },
    ),
    (
        {
            # RegNetY, different stem
            "name": "regnet",
            "depth": 22,
            "w_0": 24,
            "w_a": 24.48,
            "w_m": 2.54,
            "group_width": 16,
            "stem_type": "res_stem_in",  # optional
        },
    ),
    (
        {
            # Default minimal param set
            "name": "regnet",
            "depth": 17,
            "w_0": 192,
            "w_a": 76.82,
            "w_m": 2.19,
            "group_width": 56,
        },
    ),
]


REGNET_TEST_PRESET_NAMES = [
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1.6gf",
    "regnet_y_3.2gf",
    "regnet_y_8gf",
    "regnet_y_16gf",
    "regnet_y_32gf",
    "regnet_y_64gf",
    "regnet_y_128gf",
    "regnet_y_256gf",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_1.6gf",
    "regnet_x_3.2gf",
    "regnet_x_8gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
]

REGNET_TEST_PRESETS = [({"name": n},) for n in REGNET_TEST_PRESET_NAMES]


class TestRegNetModelBuild(unittest.TestCase):
    @parameterized.expand(REGNET_TEST_CONFIGS + REGNET_TEST_PRESETS)
    def test_build_model(self, config):
        """
        Test that the model builds using a config using either model_params or
        model_name.
        """
        model = build_model(config)
        assert isinstance(model, RegNet)
        assert model.model_depth  # Check that this attribute is properly implemented


class TestRegNetModelFW(unittest.TestCase):
    @parameterized.expand(
        [({"name": n},) for n in ["regnet_y_400mf", "regnet_x_400mf"]]
    )
    def test_model_forward(self, config):
        """
        Test that a default forward pass succeeds and does something
        """
        image_shape = (3, 224, 224)
        num_images = (10,)
        input_tensor = torch.randn(num_images + image_shape)

        model = build_model(config)
        output = model.forward(input_tensor)
        # Just check that this tensor actually went through a forward
        # pass of sorts, and was not somehow bounced back
        logging.info(f"Model {config}: output shape {output.shape}")

        assert output.shape[0] == num_images[0]

        # Default presets output 7x7 feature maps for 224x224 inputs
        assert output.shape[-1] == 7
        assert output.shape[-2] == 7
