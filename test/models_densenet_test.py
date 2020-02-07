#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.utils import compare_model_state

import torch
from classy_vision.models import build_model


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
    }
}


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

    def test_small_densenet(self):
        self._test_model(MODELS["small_densenet"])
