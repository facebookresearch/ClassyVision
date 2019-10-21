#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_args, get_test_task_config
from test.generic.utils import compare_model_state, compare_samples, compare_states

from classy_vision.hooks import LossLrMeterLoggingHook
from classy_vision.tasks import build_task


class TestClassyState(unittest.TestCase):
    def _compare_model_state(self, model_state_1, model_state_2, check_heads=True):
        compare_model_state(self, model_state_1, model_state_2, check_heads)

    def _compare_samples(self, sample_1, sample_2):
        compare_samples(self, sample_1, sample_2)

    def _compare_states(self, state_1, state_2, check_heads=True):
        compare_states(self, state_1, state_2)

    def test_get_set_state(self):
        """
        Tests the {set, get}_classy_state methods by running train_steps
        to make sure the train_steps run the same way.
        """
        config = get_test_task_config()
        # use a batchsize of 1 for faster testing
        for split in ["train", "test"]:
            config["dataset"][split]["batchsize_per_replica"] = 1
        args = get_test_args()
        task = build_task(config, args).set_hooks([LossLrMeterLoggingHook()])
        task_2 = build_task(config, args).set_hooks([LossLrMeterLoggingHook()])

        task.prepare()
        task_2.prepare()

        use_gpu = False
        local_variables = {}

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

    def test_freeze_trunk(self):
        """
        Tests that the freeze_trunk setting works as expected
        """
        config = get_test_task_config()
        config["model"]["freeze_trunk"] = True
        # use a batchsize of 1 for faster testing
        for split in ["train", "test"]:
            config["dataset"][split]["batchsize_per_replica"] = 1
        args = get_test_args()
        task = build_task(config, args, local_rank=0)

        task.prepare()

        use_gpu = False
        local_variables = {}

        # test in both train and test mode
        for i in range(2):
            task.advance_phase()

            previous_state_dict = task.get_classy_state(deep_copy=True)

            # test that after the train step the trunk remains unchanged
            task.train_step(use_gpu, local_variables)

            # compares only trunk before and after train_step
            # and should be true because the trunk is frozen
            self._compare_model_state(
                task.get_classy_state()["base_model"],
                previous_state_dict["base_model"],
                check_heads=False,
            )

            # compares both trunk and heads before and after train_step
            # and should raise AssertionError in train mode because heads
            # are training, not during test mode
            if i == 0:
                with self.assertRaises(AssertionError):
                    self._compare_model_state(
                        task.get_classy_state()["base_model"],
                        previous_state_dict["base_model"],
                        check_heads=True,
                    )
            else:
                self._compare_model_state(
                    task.get_classy_state()["base_model"],
                    previous_state_dict["base_model"],
                    check_heads=True,
                )
