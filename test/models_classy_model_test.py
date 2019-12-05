#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from test.generic.config_utils import get_test_task_config

import torch
import torch.nn as nn
from classy_vision.generic.util import load_checkpoint
from classy_vision.hooks import CheckpointHook
from classy_vision.models import ClassyModel, register_model
from classy_vision.tasks import build_task


@register_model("my_test_model")
class MyTestModel(ClassyModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    @classmethod
    def from_config(cls, config):
        return cls()


class TestClassyModel(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.base_dir)

    def test_from_checkpoint(self):
        config = get_test_task_config()
        config["model"] = {"name": "my_test_model"}
        task = build_task(config)
        task.prepare()

        local_variables = {}
        checkpoint_folder = self.base_dir + "/checkpoint_end_test/"
        input_args = {"config": config}

        # Simulate training by setting the model parameters to zero
        for param in task.model.parameters():
            param.data.zero_()

        checkpoint_hook = CheckpointHook(
            checkpoint_folder, input_args, phase_types=["train"]
        )

        # Create checkpoint dir, save checkpoint
        os.mkdir(checkpoint_folder)
        checkpoint_hook.on_start(task, local_variables)

        task.train = True
        checkpoint_hook.on_phase_end(task, local_variables)

        # Model should be checkpointed. load and compare
        checkpoint = load_checkpoint(checkpoint_folder)

        model = ClassyModel.from_checkpoint(checkpoint)
        self.assertTrue(isinstance(model, MyTestModel))

        # All parameters must be zero
        for param in model.parameters():
            self.assertTrue(torch.all(param.data == 0))
