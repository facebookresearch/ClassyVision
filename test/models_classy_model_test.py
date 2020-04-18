#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import shutil
import tempfile
import unittest
from test.generic.config_utils import get_fast_test_task_config, get_test_task_config

import torch
import torch.nn as nn
from classy_vision.generic.util import load_checkpoint
from classy_vision.heads import FullyConnectedHead
from classy_vision.hooks import CheckpointHook
from classy_vision.models import ClassyModel, ClassyModelWrapper, register_model
from classy_vision.models.classy_model import _ClassyModelAdapter
from classy_vision.tasks import build_task
from classy_vision.trainer import LocalTrainer
from torchvision import models


@register_model("my_test_model")
class MyTestModel(ClassyModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 10)

    def forward(self, x):
        return self.linear2(self.linear(x))

    @classmethod
    def from_config(cls, config):
        return cls()


class MyTestModel2(ClassyModel):
    def forward(self, x):
        return x + 1

    # need to define these properties to make the model torchscriptable
    @property
    def input_shape(self):
        return (1, 2, 3)

    @property
    def output_shape(self):
        return (4, 5, 6)

    @property
    def model_depth(self):
        return 1


class TestSimpleClassyModelWrapper(ClassyModelWrapper):
    def forward(self, x):
        return self.classy_model(x) * 2


class TestClassyModel(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.base_dir)

    def get_model_config(self, use_head):
        config = {"name": "my_test_model"}
        if use_head:
            config["heads"] = [
                {
                    "name": "fully_connected",
                    "unique_id": "default_head",
                    "num_classes": 3,
                    "fork_block": "linear",
                    "in_plane": 5,
                }
            ]
        return config

    def test_from_checkpoint(self):
        config = get_test_task_config()
        for use_head in [True, False]:
            config["model"] = self.get_model_config(use_head)
            task = build_task(config)
            task.prepare()

            checkpoint_folder = f"{self.base_dir}/{use_head}/"
            input_args = {"config": config}

            # Simulate training by setting the model parameters to zero
            for param in task.model.parameters():
                param.data.zero_()

            checkpoint_hook = CheckpointHook(
                checkpoint_folder, input_args, phase_types=["train"]
            )

            # Create checkpoint dir, save checkpoint
            os.mkdir(checkpoint_folder)
            checkpoint_hook.on_start(task)

            task.train = True
            checkpoint_hook.on_phase_end(task)

            # Model should be checkpointed. load and compare
            checkpoint = load_checkpoint(checkpoint_folder)

            model = ClassyModel.from_checkpoint(checkpoint)
            self.assertTrue(isinstance(model, MyTestModel))

            # All parameters must be zero
            for param in model.parameters():
                self.assertTrue(torch.all(param.data == 0))

    def test_classy_model_wrapper_instance(self):
        orig_wrapper_cls = MyTestModel.wrapper_cls

        # Test that we return a ClassyModel without a wrapper_cls
        MyTestModel.wrapper_cls = None
        model = MyTestModel()
        self.assertEqual(type(model), MyTestModel)
        self.assertIsInstance(model, MyTestModel)
        self.assertIsInstance(model, ClassyModel)
        self.assertIsInstance(model, nn.Module)

        # Test that we return a ClassyModelWrapper when specified as the wrapper_cls
        # The object should still pass the insinstance check
        MyTestModel.wrapper_cls = ClassyModelWrapper
        model = MyTestModel()
        self.assertEqual(type(model), ClassyModelWrapper)
        self.assertIsInstance(model, MyTestModel)
        self.assertIsInstance(model, ClassyModel)
        self.assertIsInstance(model, nn.Module)

        # restore the original wrapper class
        MyTestModel2.wrapper_cls = orig_wrapper_cls

    def test_classy_model_wrapper_torch_scriptable(self):
        orig_wrapper_cls = MyTestModel2.wrapper_cls
        input = torch.ones((2, 2))

        for wrapper_cls, expected_output in [
            (None, input + 1),
            # this isn't supported yet
            # (TestSimpleClassyModelWrapper, (input + 1) * 2),
        ]:
            MyTestModel2.wrapper_cls = wrapper_cls
            model = MyTestModel2()
            scripted_model = torch.jit.script(model)
            self.assertTrue(torch.allclose(expected_output, model(input)))
            self.assertTrue(torch.allclose(expected_output, scripted_model(input)))

        # restore the original wrapper class
        MyTestModel2.wrapper_cls = orig_wrapper_cls

    def test_classy_model_wrapper_torch_jittable(self):
        orig_wrapper_cls = MyTestModel2.wrapper_cls
        input = torch.ones((2, 2))

        for wrapper_cls, expected_output in [
            (None, input + 1),
            (TestSimpleClassyModelWrapper, (input + 1) * 2),
        ]:
            MyTestModel2.wrapper_cls = wrapper_cls
            model = MyTestModel2()
            jitted_model = torch.jit.trace(model, input)
            self.assertTrue(torch.allclose(expected_output, model(input)))
            self.assertTrue(torch.allclose(expected_output, jitted_model(input)))

        # restore the original wrapper class
        MyTestModel2.wrapper_cls = orig_wrapper_cls


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def extract_features(self, x):
        return torch.cat([x, x], dim=1)


class TestClassyModelAdapter(unittest.TestCase):
    def test_classy_model_adapter(self):
        model = TestModel()
        classy_model = ClassyModel.from_model(model)
        # test that the returned object is an instance of ClassyModel
        self.assertIsInstance(classy_model, ClassyModel)
        # test that the returned object is also an instance of _ClassyModelAdapter
        self.assertIsInstance(classy_model, _ClassyModelAdapter)

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

    def test_classy_model_adapter_properties(self):
        # test that the properties work correctly when passed to the adapter
        model = TestModel()
        num_classes = 5
        input_shape = (10,)
        output_shape = (num_classes,)
        model_depth = 1
        classy_model = ClassyModel.from_model(
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
        classy_model = ClassyModel.from_model(model)

        config = get_fast_test_task_config()
        task = build_task(config)
        task.set_model(classy_model)
        trainer = LocalTrainer()
        trainer.train(task)

    def test_heads(self):
        model = models.resnet50(pretrained=False)
        classy_model = ClassyModel.from_model(model)
        num_classes = 5
        head = FullyConnectedHead(
            unique_id="default", in_plane=2048, num_classes=num_classes
        )
        classy_model.set_heads({"layer4": [head]})
        input = torch.ones((1, 3, 224, 224))
        self.assertEqual(classy_model(input).shape, (1, num_classes))
