#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import unittest

import torch
import torch.nn as nn
from classy_vision.generic.util import get_torch_version
from classy_vision.models import build_model, RegNet
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
    (
        {
            # RegNetZ
            "name": "regnet",
            "block_type": "res_bottleneck_linear_block",
            "depth": 21,
            "w_0": 16,
            "w_a": 10.7,
            "w_m": 2.51,
            "group_width": 4,
            "bot_mul": 4.0,
            "activation": "silu",
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
    "regnet_z_500mf",
    "regnet_z_4gf",
]

REGNET_TEST_PRESETS = [({"name": n},) for n in REGNET_TEST_PRESET_NAMES]


class TestRegNetModelBuild(unittest.TestCase):
    @parameterized.expand(REGNET_TEST_CONFIGS + REGNET_TEST_PRESETS)
    def test_build_model(self, config):
        """
        Test that the model builds using a config using either model_params or
        model_name.
        """
        if get_torch_version() < [1, 7] and (
            "regnet_z" in config["name"] or config.get("activation", "relu") == "silu"
        ):
            self.skipTest("SiLU activation is only supported since PyTorch 1.7")
        model = build_model(config)
        assert isinstance(model, RegNet)

    @parameterized.expand(REGNET_TEST_CONFIGS + REGNET_TEST_PRESETS)
    def test_quantize_model(self, config):
        """
        Test that the model builds using a config using either model_params or
        model_name and calls fx graph mode quantization apis
        """
        if get_torch_version() < [1, 13]:
            self.skipTest(
                "This test is using a new api of FX Graph Mode Quantization which is only available after 1.13"
            )
        import torch.ao.quantization as tq
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        model = build_model(config)
        assert isinstance(model, RegNet)
        model.eval()
        example_inputs = (torch.rand(1, 3, 3, 3),)
        model.stem = prepare_fx(model.stem, {"": tq.default_qconfig}, example_inputs)
        model.stem = convert_fx(model.stem)


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


class TestRegNet(unittest.TestCase):
    def _compare_models(self, model_1, model_2, expect_same: bool):
        if expect_same:
            self.assertMultiLineEqual(repr(model_1), repr(model_2))
        else:
            self.assertNotEqual(repr(model_1), repr(model_2))

    def swap_relu_with_silu(self, module):
        for child_name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, child_name, nn.SiLU())
            else:
                self.swap_relu_with_silu(child)

    def _check_no_module_cls_in_model(self, module_cls, model):
        for module in model.modules():
            self.assertNotIsInstance(module, module_cls)

    @unittest.skipIf(
        get_torch_version() < [1, 7],
        "SiLU activation is only supported since PyTorch 1.7",
    )
    def test_activation(self):
        config = REGNET_TEST_CONFIGS[0][0]
        model_default = build_model(config)
        config = copy.deepcopy(config)
        config["activation"] = "relu"
        model_relu = build_model(config)

        # both models should be the same
        self._compare_models(model_default, model_relu, expect_same=True)

        # we don't expect any silus in the model
        self._check_no_module_cls_in_model(nn.SiLU, model_relu)

        config["activation"] = "silu"
        model_silu = build_model(config)

        # the models should be different
        self._compare_models(model_silu, model_relu, expect_same=False)

        # swap out all relus with silus
        self.swap_relu_with_silu(model_relu)
        print(model_relu)
        # both models should be the same
        self._compare_models(model_relu, model_silu, expect_same=True)

        # we don't expect any relus in the model
        self._check_no_module_cls_in_model(nn.ReLU, model_relu)
