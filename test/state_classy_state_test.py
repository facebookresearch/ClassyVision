#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_args, get_test_task_config
from test.generic.utils import compare_model_state, compare_samples

from classy_vision.generic.classy_trainer_common import train_step
from classy_vision.generic.util import update_classy_state
from classy_vision.hooks import LossLrMeterLoggingHook
from classy_vision.tasks import build_task


class TestClassyState(unittest.TestCase):
    def _compare_model_state(self, model_state_1, model_state_2, check_heads=True):
        compare_model_state(self, model_state_1, model_state_2, check_heads)

    def _compare_samples(self, sample_1, sample_2):
        compare_samples(self, sample_1, sample_2)

    def _compare_states(self, state_1, state_2):
        """
        Tests the classy state dicts for equality, but skips the member objects
        which implement their own {get, set}_classy_state functions.
        """
        # check base_model
        self._compare_model_state(state_1["base_model"], state_2["base_model"])
        # check losses
        self.assertEqual(len(state_1["losses"]), len(state_2["losses"]))
        for loss_1, loss_2 in zip(state_1["losses"], state_2["losses"]):
            self.assertAlmostEqual(loss_1, loss_2)

        for key in ["base_model", "meters", "optimizer", "losses"]:
            # we trust that these have been tested using their unit tests or
            # by the code above
            self.assertIn(key, state_1)
            self.assertIn(key, state_2)
            del state_1[key]
            del state_2[key]
        self.assertDictEqual(state_1, state_2)

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

        state = task.build_initial_state()
        state_2 = task_2.build_initial_state()

        use_gpu = False
        local_variables = {}

        # test in both train and test mode
        for _ in range(2):
            state.advance_phase()

            # state 2 should have the same state
            state_2.set_classy_state(state.get_classy_state(deep_copy=True))
            self._compare_states(state.get_classy_state(), state_2.get_classy_state())

            # this tests that both states' iterators return the same samples
            sample = next(state.get_data_iterator())
            sample_2 = next(state_2.get_data_iterator())
            self._compare_samples(sample, sample_2)

            # test that the train step runs the same way on both states
            # and the loss remains the same
            train_step(state, use_gpu, local_variables)
            train_step(state_2, use_gpu, local_variables)
            self._compare_states(state.get_classy_state(), state_2.get_classy_state())

    def test_update_state(self):
        """
        Tests that the update_classy_state successfully updates from a
        checkpoint
        """
        config = get_test_task_config()
        config["model"]["freeze_trunk"] = False
        # use a batchsize of 1 for faster testing
        for split in ["train", "test"]:
            config["dataset"][split]["batchsize_per_replica"] = 1
        args = get_test_args()
        task = build_task(config, args, local_rank=0)

        for reset_heads in [True, False]:
            config["reset_heads"] = reset_heads

            state = task.build_initial_state()
            state_2 = task.build_initial_state()
            # test in both train and test mode
            for _ in range(2):
                state.advance_phase()

                update_classy_state(
                    state_2, state.get_classy_state(deep_copy=True), reset_heads
                )
                self._compare_model_state(
                    state.get_classy_state()["base_model"],
                    state_2.get_classy_state()["base_model"],
                    not reset_heads,
                )

    def test_freeze_trunk(self):
        """
        Tests that the freeze_trunk setting works as expected
        """
        config = get_test_task_config()
        config["model"]["freeze_trunk"] = True
        config["reset_heads"] = True
        # use a batchsize of 1 for faster testing
        for split in ["train", "test"]:
            config["dataset"][split]["batchsize_per_replica"] = 1
        args = get_test_args()
        task = build_task(config, args, local_rank=0)

        state = task.build_initial_state()

        use_gpu = False
        local_variables = {}

        # test in both train and test mode
        for i in range(2):
            state.advance_phase()

            previous_state_dict = state.get_classy_state(deep_copy=True)

            # test that after the train step the trunk remains unchanged
            train_step(state, use_gpu, local_variables)

            # compares only trunk before and after train_step
            # and should be true because the trunk is frozen
            self._compare_model_state(
                state.get_classy_state()["base_model"],
                previous_state_dict["base_model"],
                check_heads=False,
            )

            # compares both trunk and heads before and after train_step
            # and should raise AssertionError in train mode because heads
            # are training, not during test mode
            if i == 0:
                with self.assertRaises(AssertionError):
                    self._compare_model_state(
                        state.get_classy_state()["base_model"],
                        previous_state_dict["base_model"],
                        check_heads=True,
                    )
            else:
                self._compare_model_state(
                    state.get_classy_state()["base_model"],
                    previous_state_dict["base_model"],
                    check_heads=True,
                )
