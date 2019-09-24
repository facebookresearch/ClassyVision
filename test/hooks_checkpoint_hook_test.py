#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from test.generic.config_utils import get_test_args, get_test_task_config

from classy_vision.generic.util import load_checkpoint
from classy_vision.hooks import CheckpointHook
from classy_vision.tasks import setup_task


class TestCheckpointHook(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.base_dir)

    def test_state_checkpointing(self) -> None:
        """
        Test that the state gets checkpointed without any errors, but only on the
        right phase_type and only if the checkpoint directory exists.
        """
        config = get_test_task_config()
        args = get_test_args()
        task = setup_task(config, args)
        state = task.build_initial_state()

        local_variables = {}
        checkpoint_folder = self.base_dir + "/checkpoint_end_test/"
        device = "cpu"

        # create a checkpoint hook
        checkpoint_hook = CheckpointHook(checkpoint_folder, args, phase_types=["train"])

        # checkpoint directory doesn't exist
        # call the on start function
        with self.assertRaises(FileNotFoundError):
            checkpoint_hook.on_start(state, local_variables)
        # call the on end phase function
        with self.assertRaises(AssertionError):
            checkpoint_hook.on_phase_end(state, local_variables)
        # try loading a non-existent checkpoint
        checkpoint = load_checkpoint(checkpoint_folder, device)
        self.assertIsNone(checkpoint)

        # create checkpoint dir, verify on_start hook runs
        os.mkdir(checkpoint_folder)
        checkpoint_hook.on_start(state, local_variables)

        # Phase_type is test, expect no checkpoint
        state.train = False
        # call the on end phase function
        checkpoint_hook.on_phase_end(state, local_variables)
        checkpoint = load_checkpoint(checkpoint_folder, device)
        self.assertIsNone(checkpoint)

        state.train = True
        # call the on end phase function
        checkpoint_hook.on_phase_end(state, local_variables)
        # model should be checkpointed. load and compare
        checkpoint = load_checkpoint(checkpoint_folder, device)
        self.assertIsNotNone(checkpoint)
        for key in ["input_args", "config", "classy_state_dict"]:
            self.assertIn(key, checkpoint)
        # not testing for equality of classy_state_dict, that is tested in
        # a separate test
        self.assertEqual(checkpoint["input_args"], args)
        self.assertDictEqual(checkpoint["config"], state.task.get_config())

    def test_checkpoint_period(self) -> None:
        """
        Test that the checkpoint_period works as expected.
        """
        config = get_test_task_config()
        args = get_test_args()
        task = setup_task(config, args)
        state = task.build_initial_state()

        local_variables = {}
        checkpoint_folder = self.base_dir + "/checkpoint_end_test/"
        device = "cpu"
        checkpoint_period = 10

        for phase_types in [["train"], ["train", "test"]]:
            # create a checkpoint hook
            checkpoint_hook = CheckpointHook(
                checkpoint_folder,
                args,
                phase_types=phase_types,
                checkpoint_period=checkpoint_period,
            )

            # create checkpoint dir
            os.mkdir(checkpoint_folder)

            # call the on start function
            checkpoint_hook.on_start(state, local_variables)

            # shouldn't create any checkpoints until there are checkpoint_period
            # phases which are in phase_types
            count = 0
            valid_phase_count = 0
            while valid_phase_count < checkpoint_period - 1:
                state.train = count % 2 == 0
                # call the on end phase function
                checkpoint_hook.on_phase_end(state, local_variables)
                checkpoint = load_checkpoint(checkpoint_folder, device)
                self.assertIsNone(checkpoint)
                valid_phase_count += 1 if state.phase_type in phase_types else 0
                count += 1

            # create a phase which is in phase_types
            state.train = True
            # call the on end phase function
            checkpoint_hook.on_phase_end(state, local_variables)
            # model should be checkpointed. load and compare
            checkpoint = load_checkpoint(checkpoint_folder, device)
            self.assertIsNotNone(checkpoint)
            # delete the checkpoint dir
            shutil.rmtree(checkpoint_folder)
