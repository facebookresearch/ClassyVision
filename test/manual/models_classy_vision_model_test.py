#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from collections import defaultdict
from test.generic.config_utils import get_test_model_configs
from test.generic.utils import compare_model_state

import torch
from classy_vision.heads import build_head
from classy_vision.models import ClassyModel, build_model


class TestClassyModel(unittest.TestCase):
    model_configs = get_test_model_configs()

    def _get_config(self, model_config):
        return {
            "name": "classification_task",
            "num_epochs": 12,
            "loss": {"name": "test_loss"},
            "dataset": {
                "name": "imagenet",
                "batchsize_per_replica": 8,
                "use_pairs": False,
                "num_samples_per_phase": None,
                "use_shuffle": {"train": True, "test": False},
            },
            "meters": [],
            "model": model_config,
            "optimizer": {"name": "test_opt"},
        }

    def _compare_model_state(self, state, state2):
        compare_model_state(self, state, state2)

    def test_build_model(self):
        for cfg in self.model_configs:
            config = self._get_config(cfg)
            model = build_model(config["model"])
            self.assertTrue(isinstance(model, ClassyModel))
            self.assertTrue(
                type(model.input_shape) == tuple and len(model.input_shape) == 3
            )
            self.assertTrue(
                type(model.output_shape) == tuple and len(model.output_shape) == 2
            )
            self.assertTrue(type(model.model_depth) == int)

    def test_get_set_state(self):
        config = self._get_config(self.model_configs[0])
        model = build_model(config["model"])
        fake_input = torch.Tensor(1, 3, 224, 224).float()
        model.eval()
        state = model.get_classy_state()
        with torch.no_grad():
            output = model(fake_input)

        model2 = build_model(config["model"])
        model2.set_classy_state(state)

        # compare the states
        state2 = model2.get_classy_state()
        self._compare_model_state(state, state2)

        model2.eval()
        with torch.no_grad():
            output2 = model2(fake_input)
        self.assertTrue(torch.allclose(output, output2))

        # test deep_copy by assigning a deep copied state to model2
        # and then changing the original model's state
        state = model.get_classy_state(deep_copy=True)

        model3 = build_model(config["model"])
        state3 = model3.get_classy_state()

        # assign model2's state to model's and also re-assign model's state
        model2.set_classy_state(state)
        model.set_classy_state(state3)

        # compare the states
        state2 = model2.get_classy_state()
        self._compare_model_state(state, state2)

    def test_get_set_head_states(self):
        config = copy.deepcopy(self._get_config(self.model_configs[0]))
        head_configs = config["model"]["heads"]
        config["model"]["heads"] = []
        model = build_model(config["model"])
        trunk_state = model.get_classy_state()

        heads = defaultdict(dict)
        for head_config in head_configs:
            head = build_head(head_config)
            heads[head_config["fork_block"]][head.unique_id] = head
        model.set_heads(heads)
        model_state = model.get_classy_state()

        # the heads should be the same as we set
        self.assertEqual(len(heads), len(model.get_heads()))
        for block_name, hs in model.get_heads().items():
            self.assertEqual(hs, heads[block_name])

        model._clear_heads()
        self._compare_model_state(model.get_classy_state(), trunk_state)

        model.set_heads(heads)
        self._compare_model_state(model.get_classy_state(), model_state)
