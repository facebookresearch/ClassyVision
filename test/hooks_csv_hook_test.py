#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import shutil
import tempfile
from test.generic.config_utils import get_fast_test_task_config, get_test_task_config
from test.generic.hook_test_utils import HookTestBase

import torch
from classy_vision.generic.util import load_checkpoint
from classy_vision.hooks import CheckpointHook
from classy_vision.tasks import build_task
from classy_vision.trainer import LocalTrainer


class TestCSVHook(HookTestBase):
    def setUp(self) -> None:
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.base_dir)

    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        config = {
            "checkpoint_folder": "/test/",
            "input_args": {"foo": "bar"},
            "phase_types": ["train"],
            "checkpoint_period": 2,
        }
        invalid_config = copy.deepcopy(config)
        invalid_config["checkpoint_folder"] = 12

        self.constructor_test_helper(
            config=config,
            hook_type=CheckpointHook,
            hook_registry_name="checkpoint",
            invalid_configs=[invalid_config],
        )

    def test_checkpointing(self):
        # make checkpoint directory
        checkpoint_folder = self.base_dir + "/checkpoint/"
        os.mkdir(checkpoint_folder)

        config = get_fast_test_task_config()
        cuda_available = torch.cuda.is_available()
        task = build_task(config)

        task.prepare()

        # create a checkpoint hook
        checkpoint_hook = CheckpointHook(checkpoint_folder, {}, phase_types=["train"])

        # call the on end phase function
        checkpoint_hook.on_phase_end(task)

        # we should be able to train a task using the checkpoint on all available
        # devices
        for use_gpu in {False, cuda_available}:
            # load the checkpoint
            checkpoint = load_checkpoint(checkpoint_folder)

            # create a new task
            task = build_task(config)

            # set the checkpoint
            task._set_checkpoint_dict(checkpoint)

            task.set_use_gpu(use_gpu)

            # we should be able to run the trainer using the checkpoint
            trainer = LocalTrainer()
            trainer.train(task)
