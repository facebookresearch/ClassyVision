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
from test.generic.hook_test_utils import HookTestBase

import torch
from classy_vision.generic.util import load_checkpoint
from classy_vision.hooks import CheckpointHook
from classy_vision.tasks import build_task
from classy_vision.trainer import LocalTrainer


class TestCheckpointHook(HookTestBase):
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

    def test_state_checkpointing(self) -> None:
        """
        Test that the state gets checkpointed without any errors, but only on the
        right phase_type and only if the checkpoint directory exists.
        """
        config = get_test_task_config()
        task = build_task(config)
        task.prepare()

        checkpoint_folder = self.base_dir + "/checkpoint_end_test/"
        input_args = {"foo": "bar"}

        # create a checkpoint hook
        checkpoint_hook = CheckpointHook(
            checkpoint_folder, input_args, phase_types=["train"]
        )

        # checkpoint directory doesn't exist
        # call the on start function
        with self.assertRaises(FileNotFoundError):
            checkpoint_hook.on_start(task)
        # call the on end phase function
        with self.assertRaises(AssertionError):
            checkpoint_hook.on_phase_end(task)
        # try loading a non-existent checkpoint
        checkpoint = load_checkpoint(checkpoint_folder)
        self.assertIsNone(checkpoint)

        # create checkpoint dir, verify on_start hook runs
        os.mkdir(checkpoint_folder)
        checkpoint_hook.on_start(task)

        # Phase_type is test, expect no checkpoint
        task.train = False
        # call the on end phase function
        checkpoint_hook.on_phase_end(task)
        checkpoint = load_checkpoint(checkpoint_folder)
        self.assertIsNone(checkpoint)

        task.train = True
        # call the on end phase function
        checkpoint_hook.on_phase_end(task)
        # model should be checkpointed. load and compare
        checkpoint = load_checkpoint(checkpoint_folder)
        self.assertIsNotNone(checkpoint)
        for key in ["input_args", "classy_state_dict"]:
            self.assertIn(key, checkpoint)
        # not testing for equality of classy_state_dict, that is tested in
        # a separate test
        self.assertDictEqual(checkpoint["input_args"], input_args)

    def test_checkpoint_period(self) -> None:
        """
        Test that the checkpoint_period works as expected.
        """
        config = get_test_task_config()
        task = build_task(config)
        task.prepare()

        checkpoint_folder = self.base_dir + "/checkpoint_end_test/"
        checkpoint_period = 10

        for phase_types in [["train"], ["train", "test"]]:
            # create a checkpoint hook
            checkpoint_hook = CheckpointHook(
                checkpoint_folder,
                {},
                phase_types=phase_types,
                checkpoint_period=checkpoint_period,
            )

            # create checkpoint dir
            os.mkdir(checkpoint_folder)

            # call the on start function
            checkpoint_hook.on_start(task)

            # shouldn't create any checkpoints until there are checkpoint_period
            # phases which are in phase_types
            count = 0
            valid_phase_count = 0
            while valid_phase_count < checkpoint_period - 1:
                task.train = count % 2 == 0
                # call the on end phase function
                checkpoint_hook.on_phase_end(task)
                checkpoint = load_checkpoint(checkpoint_folder)
                self.assertIsNone(checkpoint)
                valid_phase_count += 1 if task.phase_type in phase_types else 0
                count += 1

            # create a phase which is in phase_types
            task.train = True
            # call the on end phase function
            checkpoint_hook.on_phase_end(task)
            # model should be checkpointed. load and compare
            checkpoint = load_checkpoint(checkpoint_folder)
            self.assertIsNotNone(checkpoint)
            # delete the checkpoint dir
            shutil.rmtree(checkpoint_folder)

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
            task.set_checkpoint(checkpoint)

            task.set_use_gpu(use_gpu)

            # we should be able to run the trainer using the checkpoint
            trainer = LocalTrainer()
            trainer.train(task)
