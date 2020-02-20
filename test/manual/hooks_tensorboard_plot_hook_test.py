#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import shutil
import tempfile
import unittest
import unittest.mock as mock
from itertools import product
from test.generic.config_utils import get_test_mlp_task_config, get_test_task_config

from classy_vision.hooks import TensorboardPlotHook
from classy_vision.optim.param_scheduler import UpdateInterval
from classy_vision.tasks import build_task
from classy_vision.trainer import LocalTrainer
from tensorboardX import SummaryWriter


class TestTensorboardPlotHook(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.base_dir)

    @mock.patch("classy_vision.hooks.tensorboard_plot_hook.is_master")
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
            task = build_task(config)
            task.prepare()
            task.phase_idx = phase_idx
            task.train = train

            losses = [1.23, 4.45, 12.3, 3.4]

            local_variables = {}

            summary_writer = SummaryWriter(self.base_dir)
            # create a spy on top of summary_writer
            summary_writer = mock.MagicMock(wraps=summary_writer)

            # create a loss lr tensorboard hook
            tensorboard_plot_hook = TensorboardPlotHook(summary_writer)

            # test that the hook logs a warning and doesn't write anything to
            # the writer if on_phase_start() is not called for initialization
            # before on_step() is called.
            with self.assertLogs() as log_watcher:
                tensorboard_plot_hook.on_step(task, local_variables)

            self.assertTrue(
                len(log_watcher.records) == 1
                and log_watcher.records[0].levelno == logging.WARN
                and "learning_rates is not initialized" in log_watcher.output[0]
            )

            # test that the hook logs a warning and doesn't write anything to
            # the writer if on_phase_start() is not called for initialization
            # if on_phase_end() is called.
            with self.assertLogs() as log_watcher:
                tensorboard_plot_hook.on_phase_end(task, local_variables)

            self.assertTrue(
                len(log_watcher.records) == 1
                and log_watcher.records[0].levelno == logging.WARN
                and "learning_rates is not initialized" in log_watcher.output[0]
            )
            summary_writer.add_scalar.reset_mock()

            # run the hook in the correct order
            tensorboard_plot_hook.on_phase_start(task, local_variables)

            for loss in losses:
                task.losses.append(loss)
                tensorboard_plot_hook.on_step(task, local_variables)

            tensorboard_plot_hook.on_phase_end(task, local_variables)

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
                for meter in task.meters:
                    for name in meter.value:
                        meter_key = f"{phase_type}_{meter.name}_{name}"
                        summary_writer.add_scalar.assert_any_call(
                            meter_key, mock.ANY, global_step=mock.ANY
                        )
            else:
                # add_scalar() shouldn't be called since is_master() is False
                summary_writer.add_scalar.assert_not_called()
            summary_writer.add_scalar.reset_mock()

    def test_logged_lr(self):
        # Mock LR scheduler
        def scheduler_mock(where):
            return where

        mock_lr_scheduler = mock.Mock(side_effect=scheduler_mock)
        mock_lr_scheduler.update_interval = UpdateInterval.STEP

        # Mock Logging
        class DummySummaryWriter(object):
            def __init__(self):
                self.scalar_logs = {}

            def add_scalar(self, key, value, global_step=None, walltime=None) -> None:
                self.scalar_logs[key] = self.scalar_logs.get(key, []) + [value]

            def flush(self):
                return

        config = get_test_mlp_task_config()
        config["num_epochs"] = 3
        config["dataset"]["train"]["batchsize_per_replica"] = 5
        config["dataset"]["test"]["batchsize_per_replica"] = 5
        task = build_task(config)

        writer = DummySummaryWriter()
        hook = TensorboardPlotHook(writer)
        task.set_hooks([hook])
        task.optimizer.param_schedulers["lr"] = mock_lr_scheduler

        trainer = LocalTrainer()
        trainer.train(task)

        # We have 10 samples, batch size is 5. Each epoch is done in two steps.
        self.assertEqual(
            writer.scalar_logs["train_learning_rate_updates"],
            [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6],
        )
