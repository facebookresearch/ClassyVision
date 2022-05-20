#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.generic.util import get_torch_version
from classy_vision.models import build_model
from test.generic.utils import compare_model_state


MODELS = {
    "small_densenet": {
        "name": "densenet",
        "num_blocks": [1, 1, 1, 1],
        "init_planes": 4,
        "growth_rate": 32,
        "expansion": 4,
        "final_bn_relu": True,
        "small_input": True,
        "heads": [
            {
                "name": "fully_connected",
                "unique_id": "default_head",
                "num_classes": 1000,
                "fork_block": "trunk_output",
                "in_plane": 60,
                "zero_init_bias": True,
            }
        ],
    },
    "small_densenet_se": {
        "name": "densenet",
        "num_blocks": [1, 1, 1, 1],
        "init_planes": 4,
        "growth_rate": 32,
        "expansion": 4,
        "final_bn_relu": True,
        "small_input": True,
        "use_se": True,
        "heads": [
            {
                "name": "fully_connected",
                "unique_id": "default_head",
                "num_classes": 1000,
                "fork_block": "trunk_output",
                "in_plane": 60,
                "zero_init_bias": True,
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


class TestDensenet(unittest.TestCase):
    def _test_model(self, model_config):
        """This test will build Densenet models, run a forward pass and
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
        if get_torch_version() >= [1, 11]:
            import torch.ao.quantization as tq
            from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
        else:
            import torch.quantization as tq
            from torch.quantization.quantize_fx import convert_fx, prepare_fx

        # quantize model
        model = build_model(model_config)
        model.eval()

        input = torch.ones([1, 3, 32, 32])

        heads = model.get_heads()
        # since prepare changes the code of ClassyBlock we need to clear head first
        # and reattach it later to avoid caching
        model.clear_heads()

        prepare_custom_config_dict = {}
        head_path_from_blocks = [
            _find_block_full_path(model.features, block_name)
            for block_name in heads.keys()
        ]
        # TODO[quant-example-inputs]: The dimension here is random, if we need to
        # use dimension/rank in the future we'd need to get the correct dimensions
        standalone_example_inputs = (torch.randn(1, 3, 3, 3),)
        # we need to keep the modules used in head standalone since
        # it will be accessed with path name directly in execution
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
        # TODO[quant-example-inputs]: The dimension here is random, if we need to
        # use dimension/rank in the future we'd need to get the correct dimensions
        example_inputs = (torch.randn(1, 3, 3, 3),)
        model.initial_block = prepare_fx(
            model.initial_block, {"": tq.default_qconfig}, example_inputs
        )

        model.features = prepare_fx(
            model.features,
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
        model.features = convert_fx(model.features)
        model.set_heads(heads)

        output = model(input)
        self.assertEqual(output.size(), (1, 1000))

    def test_small_densenet(self):
        self._test_model(MODELS["small_densenet"])

    @unittest.skipIf(
        get_torch_version() < [1, 13],
        "This test is using a new api of FX Graph Mode Quantization which is only available after 1.13",
    )
    def test_quantized_small_densenet(self):
        self._test_quantize_model(MODELS["small_densenet"])
