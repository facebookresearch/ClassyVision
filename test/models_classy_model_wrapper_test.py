#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from test.generic.config_utils import get_fast_test_task_config

import torch
import torch.nn as nn
from classy_vision.models import ClassyModel
from classy_vision.models.classy_model_wrapper import ClassyModelWrapper
from classy_vision.tasks import build_task
from classy_vision.trainer import LocalTrainer
from torchvision import models


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def extract_features(self, x):
        return torch.cat([x, x], dim=1)


class TestClassyModelWrapper(unittest.TestCase):
    def test_classy_model_wrapper(self):
        model = TestModel()
        classy_model = ClassyModelWrapper(model)
        # test that the returned object is an instance of ClassyModel
        self.assertIsInstance(classy_model, ClassyModel)

        # test that forward works correctly
        input = torch.zeros((100, 10))
        output = classy_model(input)
        self.assertEqual(output.shape, (100, 5))

        # test that extract_features works correctly
        input = torch.zeros((100, 10))
        output = classy_model.extract_features(input)
        self.assertEqual(output.shape, (100, 20))

        # test that get_classy_state and set_classy_state work
        nn.init.constant_(classy_model.model.linear.weight, 1)
        weights = copy.deepcopy(classy_model.model.linear.weight.data)
        state_dict = classy_model.get_classy_state(deep_copy=True)
        nn.init.constant_(classy_model.model.linear.weight, 0)
        classy_model.set_classy_state(state_dict)
        self.assertTrue(torch.allclose(weights, classy_model.model.linear.weight.data))

    def test_classy_model_wrapper_properties(self):
        # test that the properties work correctly when passed to the wrapper
        model = TestModel()
        num_classes = 5
        input_shape = (10,)
        output_shape = (num_classes,)
        model_depth = 1
        classy_model = ClassyModelWrapper(
            model,
            input_shape=input_shape,
            output_shape=output_shape,
            model_depth=model_depth,
        )
        self.assertEqual(classy_model.input_shape, input_shape)
        self.assertEqual(classy_model.output_shape, output_shape)
        self.assertEqual(classy_model.model_depth, model_depth)

    def test_train_step(self):
        # test that the model can be run in a train step
        model = models.resnet34(pretrained=False)
        classy_model = ClassyModelWrapper(model)

        config = get_fast_test_task_config()
        task = build_task(config)
        task.set_model(classy_model)
        trainer = LocalTrainer()
        trainer.train(task)
