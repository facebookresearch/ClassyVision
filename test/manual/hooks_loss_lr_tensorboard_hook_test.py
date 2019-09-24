#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import shutil
import tempfile
import unittest
import unittest.mock as mock
from itertools import product
from test.generic.config_utils import get_test_args, get_test_task_config

from classy_vision.hooks import LossLrTensorboardHook
from classy_vision.tasks import setup_task
from tensorboardX import SummaryWriter


class TestLossLrTensorboardHook(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.base_dir)

    @mock.patch("classy_vision.hooks.loss_lr_tensorboard_hook.is_master")
    def test_writer(self, mock_is_master_func: mock.MagicMock) -> None:
        """
        Tests that the tensorboard writer writes the correct scalars to SummaryWriter
        iff is_master() is True.
        """
        for phase_idx, master in product([0, 1, 2], [True, False]):
            train, phase_type = (
                (True, "train") if phase_idx % 2 == 0 else (False, "test")
            )
            mock_is_master_func.return_value = master

            # set up the task and state
            config = get_test_task_config()
            config["dataset"]["train"]["batchsize_per_replica"] = 2
            config["dataset"]["test"]["batchsize_per_replica"] = 5
            args = get_test_args()
            task = setup_task(config, args)
            state = task.build_initial_state()
            state.phase_idx = phase_idx
            state.train = train

            losses = [1.23, 4.45, 12.3, 3.4]

            local_variables = {}

            summary_writer = SummaryWriter(self.base_dir)
            # create a spy on top of summary_writer
            summary_writer = mock.MagicMock(wraps=summary_writer)

            # create a loss lr tensorboard hook
            loss_lr_tensorboard_hook = LossLrTensorboardHook(summary_writer)

            # test that the hook logs a warning and doesn't write anything to
            # the writer if on_phase_start() is not called for initialization
            # before on_loss() is called.
            with self.assertLogs() as log_watcher:
                loss_lr_tensorboard_hook.on_loss(state, local_variables)

            self.assertTrue(
                len(log_watcher.records) == 1
                and log_watcher.records[0].levelno == logging.WARN
                and "learning_rates is not initialized" in log_watcher.output[0]
            )

            # test that the hook logs a warning and doesn't write anything to
            # the writer if on_phase_start() is not called for initialization
            # if on_phase_end() is called.
            with self.assertLogs() as log_watcher:
                loss_lr_tensorboard_hook.on_phase_end(state, local_variables)

            self.assertTrue(
                len(log_watcher.records) == 1
                and log_watcher.records[0].levelno == logging.WARN
                and "learning_rates is not initialized" in log_watcher.output[0]
            )
            summary_writer.add_scalar.reset_mock()

            # run the hook in the correct order
            loss_lr_tensorboard_hook.on_phase_start(state, local_variables)

            for loss in losses:
                state.losses.append(loss)
                loss_lr_tensorboard_hook.on_loss(state, local_variables)

            loss_lr_tensorboard_hook.on_phase_end(state, local_variables)

            if master:
                # add_scalar() should have been called with the right scalars
                if train:
                    loss_key = f"{phase_type}_loss"
                    learning_rate_key = f"{phase_type}_learning_rate_updates"
                    summary_writer.add_scalar.assert_any_call(
                        loss_key, mock.ANY, global_step=mock.ANY, walltime=mock.ANY
                    )
                    summary_writer.add_scalar.assert_any_call(
                        learning_rate_key,
                        mock.ANY,
                        global_step=mock.ANY,
                        walltime=mock.ANY,
                    )
                avg_loss_key = f"avg_{phase_type}_loss"
                summary_writer.add_scalar.assert_any_call(
                    avg_loss_key, mock.ANY, global_step=mock.ANY
                )
            else:
                # add_scalar() shouldn't be called since is_master() is False
                summary_writer.add_scalar.assert_not_called()
            summary_writer.add_scalar.reset_mock()
