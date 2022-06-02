#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import unittest

import torch
import torchvision.models
from classy_vision.generic.util import get_torch_version
from classy_vision.models import build_model, ResNeXt
from test.generic.utils import compare_model_state


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


def _find_block_full_path(model, block_name):
    """Find the full path for a given block name
    e.g. block3-1 --> 3.block3-1
    """
    for name, _ in model.named_modules():
        if name.endswith(block_name):
            return name
    return None


def _post_training_quantize(model, input):
    if get_torch_version() >= [1, 11]:
        import torch.ao.quantization as tq
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    else:
        import torch.quantization as tq
        from torch.quantization.quantize_fx import convert_fx, prepare_fx

    model.eval()
    # running
    model(*(input,))
    fqn_to_example_inputs = None
    if get_torch_version() >= [1, 13]:
        from torch.ao.quantization.utils import get_fqn_to_example_inputs

        fqn_to_example_inputs = get_fqn_to_example_inputs(model, (input,))

    heads = model.get_heads()
    # since prepare changes the code of ClassyBlock we need to clear head first
    # and reattach it later to avoid caching
    model.clear_heads()

    prepare_custom_config_dict = {}
    head_path_from_blocks = [
        _find_block_full_path(model.blocks, block_name) for block_name in heads.keys()
    ]

    # we need to keep the modules used in head standalone since
    # it will be accessed with path name directly in execution
    if get_torch_version() >= [1, 13]:
        prepare_custom_config_dict["standalone_module_name"] = [
            (
                head,
                {"": tq.default_qconfig},
                fqn_to_example_inputs["blocks." + head],
                {"input_quantized_idxs": [0], "output_quantized_idxs": []},
                None,
            )
            for head in head_path_from_blocks
        ]
    else:
        standalone_example_inputs = (torch.rand(1, 3, 3, 3),)
        prepare_custom_config_dict["standalone_module_name"] = [
            (
                head,
                {"": tq.default_qconfig},
                standalone_example_inputs,
                {"input_quantized_idxs": [0], "output_quantized_idxs": []},
                None,
            )
            for head in head_path_from_blocks
        ]
    example_inputs = (torch.rand(1, 3, 3, 3),)
    if get_torch_version() >= [1, 13]:
        example_inputs = fqn_to_example_inputs["initial_block"]
    model.initial_block = prepare_fx(
        model.initial_block, {"": tq.default_qconfig}, example_inputs
    )

    if get_torch_version() >= [1, 13]:
        example_inputs = fqn_to_example_inputs["blocks"]
    model.blocks = prepare_fx(
        model.blocks,
        {"": tq.default_qconfig},
        example_inputs,
        prepare_custom_config_dict,
    )
    model.set_heads(heads)

    # calibration
    model(input)

    heads = model.get_heads()
    model.clear_heads()
    model.initial_block = convert_fx(model.initial_block)
    model.blocks = convert_fx(model.blocks)
    model.set_heads(heads)
    return model


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

    def _test_quantize_model(self, model_config):
        """This test will build ResNeXt-* models, quantize the model
        with fx graph mode quantization, run a forward pass and
        verify output shape, and then verify that get / set state
        works.
        """
        model = build_model(model_config)
        # Verify forward pass works
        input = torch.ones([1, 3, 32, 32])
        output = model.forward(input)
        self.assertEqual(output.size(), (1, 1000))

        model = _post_training_quantize(model, input)

        # Verify forward pass works
        input = torch.ones([1, 3, 32, 32])
        output = model.forward(input)
        self.assertEqual(output.size(), (1, 1000))

        # Verify get_set_state
        new_model = build_model(model_config)
        new_model = _post_training_quantize(new_model, input)
        state = model.get_classy_state()
        new_model.set_classy_state(state)
        # TODO: test get state for new_model and make sure
        # it is the same as state,
        # Currently allclose is not supported in quantized tensors
        # so we can't check this right now

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

    @unittest.skipIf(
        get_torch_version() < [1, 13],
        "This test is using a new api of FX Graph Mode Quantization which is only available after 1.13",
    )
    def test_quantized_small_resnext(self):
        self._test_quantize_model(MODELS["small_resnext"])

    def test_small_resnet(self):
        self._test_model(MODELS["small_resnet"])

    @unittest.skipIf(
        get_torch_version() < [1, 13],
        "This test is using a new api of FX Graph Mode Quantization which is only available after 1.13",
    )
    def test_quantized_small_resnet(self):
        self._test_quantize_model(MODELS["small_resnet"])

    def test_small_resnet_se(self):
        self._test_model(MODELS["small_resnet_se"])

    @unittest.skipIf(
        get_torch_version() < [1, 13],
        "This test is using a new api of FX Graph Mode Quantization which is only available after 1.13",
    )
    def test_quantized_small_resnet_se(self):
        self._test_quantize_model(MODELS["small_resnet_se"])


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
