#!/usr/bin/env python3

import unittest
from test.generic.config_utils import get_test_args, get_test_task_config
from test.generic.utils import compare_model_state, compare_samples

from classy_vision.generic.classy_trainer_common import train_step
from classy_vision.tasks import setup_task


class TestClassyState(unittest.TestCase):
    def _compare_model_state(self, model_state_1, model_state_2):
        compare_model_state(self, model_state_1, model_state_2)

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
        task = setup_task(config, args, local_rank=0)

        state = task.build_initial_state()
        state_2 = task.build_initial_state()

        hooks = []
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
            train_step(state, hooks, use_gpu, local_variables)
            train_step(state_2, hooks, use_gpu, local_variables)
            self._compare_states(state.get_classy_state(), state_2.get_classy_state())
