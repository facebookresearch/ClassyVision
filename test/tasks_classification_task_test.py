#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_fast_test_task_config, get_test_task_config
from test.generic.utils import compare_model_state, compare_samples, compare_states

import torch
from classy_vision.dataset import build_dataset
from classy_vision.hooks import LossLrMeterLoggingHook
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

    def test_build_task(self):
        config = get_test_task_config()
        task = build_task(config)
        self.assertTrue(isinstance(task, ClassificationTask))

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

        task.prepare(num_dataloader_workers=1, pin_memory=False)

        task = build_task(config)
        task.prepare(num_dataloader_workers=1, pin_memory=False)

    def test_get_set_state(self):
        """
        Tests the {set, get}_classy_state methods by running train_steps
        to make sure the train_steps run the same way.
        """
        config = get_fast_test_task_config()
        task = build_task(config).set_hooks([LossLrMeterLoggingHook()])
        task_2 = build_task(config).set_hooks([LossLrMeterLoggingHook()])

        use_gpu = torch.cuda.is_available()
        local_variables = {}

        # prepare the tasks for the right device
        task.prepare(use_gpu=use_gpu)
        task_2.prepare(use_gpu=use_gpu)

        # test in both train and test mode
        for _ in range(2):
            task.advance_phase()

            # task 2 should have the same state
            task_2.set_classy_state(task.get_classy_state(deep_copy=True))
            self._compare_states(task.get_classy_state(), task_2.get_classy_state())

            # this tests that both states' iterators return the same samples
            sample = next(task.get_data_iterator())
            sample_2 = next(task_2.get_data_iterator())
            self._compare_samples(sample, sample_2)

            # test that the train step runs the same way on both states
            # and the loss remains the same
            task.train_step(use_gpu, local_variables)
            task_2.train_step(use_gpu, local_variables)
            self._compare_states(task.get_classy_state(), task_2.get_classy_state())

    @unittest.skipUnless(torch.cuda.is_available(), "This test needs a gpu to run")
    def test_get_set_state_different_devices(self):
        config = get_fast_test_task_config()
        task = build_task(config)
        task_2 = build_task(config)

        for use_gpu in [True, False]:
            task.prepare(use_gpu=use_gpu)
            task_2.prepare(use_gpu=not use_gpu)

            task_2.set_classy_state(task.get_classy_state(deep_copy=True))

            # the parameters are in different devices
            with self.assertRaises(Exception):
                self._compare_states(task.get_classy_state(), task_2.get_classy_state())

            # prepare the task for the right device
            task_2.prepare(use_gpu=use_gpu)
            self._compare_states(task.get_classy_state(), task_2.get_classy_state())

            # we should be able to run the trainer using state from a different device
            trainer = LocalTrainer(use_gpu=use_gpu)
            trainer.train(task_2)
