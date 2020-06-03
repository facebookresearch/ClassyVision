#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import shutil
import tempfile
import unittest
from test.generic.config_utils import get_fast_test_task_config, get_test_task_config
from test.generic.utils import compare_model_state, compare_samples, compare_states

import torch
from classy_vision.dataset import build_dataset
from classy_vision.generic.util import get_checkpoint_dict
from classy_vision.hooks import CheckpointHook, LossLrMeterLoggingHook
from classy_vision.losses import build_loss
from classy_vision.models import build_model
from classy_vision.optim import build_optimizer
from classy_vision.tasks import ClassificationTask, build_task
from classy_vision.trainer import LocalTrainer


class TestClassificationTask(unittest.TestCase):
    def _compare_model_state(self, model_state_1, model_state_2, check_heads=True):
        compare_model_state(self, model_state_1, model_state_2, check_heads)

    def _compare_samples(self, sample_1, sample_2):
        compare_samples(self, sample_1, sample_2)

    def _compare_states(self, state_1, state_2, check_heads=True):
        compare_states(self, state_1, state_2)

    def setUp(self):
        # create a base directory to write checkpoints to
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        # delete all the temporary data created
        shutil.rmtree(self.base_dir)

    def test_build_task(self):
        config = get_test_task_config()
        task = build_task(config)
        self.assertTrue(isinstance(task, ClassificationTask))

    def test_hooks_config_builds_correctly(self):
        config = get_test_task_config()
        config["hooks"] = [{"name": "loss_lr_meter_logging"}]
        task = build_task(config)
        self.assertTrue(len(task.hooks) == 1)
        self.assertTrue(isinstance(task.hooks[0], LossLrMeterLoggingHook))

    def test_get_state(self):
        config = get_test_task_config()
        loss = build_loss(config["loss"])
        task = (
            ClassificationTask()
            .set_num_epochs(1)
            .set_loss(loss)
            .set_model(build_model(config["model"]))
            .set_optimizer(build_optimizer(config["optimizer"]))
        )
        for phase_type in ["train", "test"]:
            dataset = build_dataset(config["dataset"][phase_type])
            task.set_dataset(dataset, phase_type)

        task.prepare()

        task = build_task(config)
        task.prepare()

    def test_synchronize_losses_non_distributed(self):
        """
        Tests that synchronize losses has no side effects in a non-distributed setting.
        """
        test_config = get_fast_test_task_config()
        task = build_task(test_config)
        task.prepare()

        old_losses = copy.deepcopy(task.losses)
        task.synchronize_losses()
        self.assertEqual(old_losses, task.losses)

    def test_synchronize_losses_when_losses_empty(self):
        config = get_fast_test_task_config()
        task = build_task(config)
        task.prepare()

        task.set_use_gpu(torch.cuda.is_available())

        # Losses should be empty when creating task
        self.assertEqual(len(task.losses), 0)

        task.synchronize_losses()

    def test_checkpointing(self):
        """
        Tests checkpointing by running train_steps to make sure the train_steps
        run the same way after loading from a checkpoint.
        """
        config = get_fast_test_task_config()
        task = build_task(config).set_hooks([LossLrMeterLoggingHook()])
        task_2 = build_task(config).set_hooks([LossLrMeterLoggingHook()])

        task.set_use_gpu(torch.cuda.is_available())

        # prepare the tasks for the right device
        task.prepare()

        # test in both train and test mode
        for _ in range(2):
            task.advance_phase()

            # set task's state as task_2's checkpoint
            task_2._set_checkpoint_dict(get_checkpoint_dict(task, {}, deep_copy=True))
            task_2.prepare()

            # task 2 should have the same state
            self._compare_states(task.get_classy_state(), task_2.get_classy_state())

            # this tests that both states' iterators return the same samples
            sample = next(task.get_data_iterator())
            sample_2 = next(task_2.get_data_iterator())
            self._compare_samples(sample, sample_2)

            # test that the train step runs the same way on both states
            # and the loss remains the same
            task.train_step()
            task_2.train_step()
            self._compare_states(task.get_classy_state(), task_2.get_classy_state())

    def test_final_train_checkpoint(self):
        """Test that a train phase checkpoint with a where of 1.0 can be loaded"""

        config = get_fast_test_task_config()
        task = build_task(config).set_hooks(
            [CheckpointHook(self.base_dir, {}, phase_types=["train"])]
        )
        task_2 = build_task(config)

        task.set_use_gpu(torch.cuda.is_available())

        trainer = LocalTrainer()
        trainer.train(task)

        # make sure fetching the where raises an exception, which means that
        # where is >= 1.0
        with self.assertRaises(Exception):
            task.where

        # set task_2's state as task's final train checkpoint
        task_2.set_checkpoint(self.base_dir)
        task_2.prepare()

        # we should be able to train the task
        trainer.train(task_2)

    def test_test_only_checkpointing(self):
        """
        Tests checkpointing by running train_steps to make sure the
        train_steps run the same way after loading from a training
        task checkpoint on a test_only task.
        """
        train_config = get_fast_test_task_config()
        train_config["num_epochs"] = 10
        test_config = get_fast_test_task_config()
        test_config["test_only"] = True
        train_task = build_task(train_config).set_hooks([LossLrMeterLoggingHook()])
        test_only_task = build_task(test_config).set_hooks([LossLrMeterLoggingHook()])

        # prepare the tasks for the right device
        train_task.prepare()

        # test in both train and test mode
        trainer = LocalTrainer()
        trainer.train(train_task)

        # set task's state as task_2's checkpoint
        test_only_task._set_checkpoint_dict(
            get_checkpoint_dict(train_task, {}, deep_copy=True)
        )
        test_only_task.prepare()
        test_state = test_only_task.get_classy_state()

        # We expect the phase idx to be different for a test only task
        self.assertEqual(test_state["phase_idx"], -1)

        # We expect that test only state is test, no matter what train state is
        self.assertFalse(test_state["train"])

        # Num updates should be 0
        self.assertEqual(test_state["num_updates"], 0)

        # train_phase_idx should -1
        self.assertEqual(test_state["train_phase_idx"], -1)

        # Verify task will run
        trainer = LocalTrainer()
        trainer.train(test_only_task)

    def test_test_only_task(self):
        """
        Tests the task in test mode by running train_steps
        to make sure the train_steps run as expected on a
        test_only task
        """
        test_config = get_fast_test_task_config()
        test_config["test_only"] = True

        # delete train dataset
        del test_config["dataset"]["train"]

        test_only_task = build_task(test_config).set_hooks([LossLrMeterLoggingHook()])

        test_only_task.prepare()
        test_state = test_only_task.get_classy_state()

        # We expect that test only state is test, no matter what train state is
        self.assertFalse(test_state["train"])

        # Num updates should be 0
        self.assertEqual(test_state["num_updates"], 0)

        # Verify task will run
        trainer = LocalTrainer()
        trainer.train(test_only_task)

    def test_train_only_task(self):
        """
        Tests that the task runs when only a train dataset is specified.
        """
        test_config = get_fast_test_task_config()

        # delete the test dataset from the config
        del test_config["dataset"]["test"]

        task = build_task(test_config).set_hooks([LossLrMeterLoggingHook()])
        task.prepare()

        # verify the the task can still be trained
        trainer = LocalTrainer()
        trainer.train(task)

    @unittest.skipUnless(torch.cuda.is_available(), "This test needs a gpu to run")
    def test_checkpointing_different_device(self):
        config = get_fast_test_task_config()
        task = build_task(config)
        task_2 = build_task(config)

        for use_gpu in [True, False]:
            task.set_use_gpu(use_gpu)
            task.prepare()

            # set task's state as task_2's checkpoint
            task_2._set_checkpoint_dict(get_checkpoint_dict(task, {}, deep_copy=True))

            # we should be able to run the trainer using state from a different device
            trainer = LocalTrainer()
            task_2.set_use_gpu(not use_gpu)
            trainer.train(task_2)
