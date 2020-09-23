#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import shutil
import tempfile
import unittest.mock as mock
from itertools import product
from test.generic.config_utils import get_test_mlp_task_config, get_test_task_config
from test.generic.hook_test_utils import HookTestBase

from classy_vision.hooks import TensorboardPlotHook
from classy_vision.optim.param_scheduler import ClassyParamScheduler, UpdateInterval
from classy_vision.tasks import build_task
from classy_vision.tasks.classification_task import LastBatchInfo
from classy_vision.trainer import LocalTrainer
from torch.utils.tensorboard import SummaryWriter


class TestTensorboardPlotHook(HookTestBase):
    def setUp(self) -> None:
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.base_dir)

    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        config = {"summary_writer": {}, "log_period": 5}
        invalid_config = copy.deepcopy(config)
        invalid_config["log_period"] = "this is not an int"

        self.constructor_test_helper(
            config=config,
            hook_type=TensorboardPlotHook,
            hook_registry_name="tensorboard_plot",
            hook_kwargs={"tb_writer": SummaryWriter(), "log_period": 5},
            invalid_configs=[invalid_config],
        )

    @mock.patch("classy_vision.hooks.tensorboard_plot_hook.is_primary")
    def test_writer(self, mock_is_primary_func: mock.MagicMock) -> None:
        """
        Tests that the tensorboard writer writes the correct scalars to SummaryWriter
        iff is_primary() is True.
        """
        for phase_idx, master in product([0, 1, 2], [True, False]):
            train, phase_type = (
                (True, "train") if phase_idx % 2 == 0 else (False, "test")
            )
            mock_is_primary_func.return_value = master

            # set up the task and state
            config = get_test_task_config()
            config["dataset"]["train"]["batchsize_per_replica"] = 2
            config["dataset"]["test"]["batchsize_per_replica"] = 5
            task = build_task(config)
            task.prepare()
            task.advance_phase()
            task.phase_idx = phase_idx
            task.train = train

            losses = [1.23, 4.45, 12.3, 3.4]
            sample_fetch_times = [1.1, 2.2, 3.3, 2.2]

            summary_writer = SummaryWriter(self.base_dir)
            # create a spy on top of summary_writer
            summary_writer = mock.MagicMock(wraps=summary_writer)

            # create a loss lr tensorboard hook
            tensorboard_plot_hook = TensorboardPlotHook(summary_writer)

            # run the hook in the correct order
            tensorboard_plot_hook.on_phase_start(task)

            # test tasks which do not pass the sample_fetch_times as well
            disable_sample_fetch_times = phase_idx == 0

            for loss, sample_fetch_time in zip(losses, sample_fetch_times):
                task.losses.append(loss)
                step_data = (
                    {}
                    if disable_sample_fetch_times
                    else {"sample_fetch_time": sample_fetch_time}
                )
                task.last_batch = LastBatchInfo(None, None, None, None, step_data)
                tensorboard_plot_hook.on_step(task)

            tensorboard_plot_hook.on_phase_end(task)

            if master:
                # add_scalar() should have been called with the right scalars
                if train:
                    learning_rate_key = f"Learning Rate/{phase_type}"
                    summary_writer.add_scalar.assert_any_call(
                        learning_rate_key,
                        mock.ANY,
                        global_step=mock.ANY,
                        walltime=mock.ANY,
                    )
                avg_loss_key = f"Losses/{phase_type}"
                summary_writer.add_scalar.assert_any_call(
                    avg_loss_key, mock.ANY, global_step=mock.ANY
                )
                for meter in task.meters:
                    for name in meter.value:
                        meter_key = f"Meters/{phase_type}/{meter.name}/{name}"
                        summary_writer.add_scalar.assert_any_call(
                            meter_key, mock.ANY, global_step=mock.ANY
                        )
                if step_data:
                    summary_writer.add_scalar.assert_any_call(
                        f"Speed/{phase_type}/cumulative_sample_fetch_time",
                        mock.ANY,
                        global_step=mock.ANY,
                        walltime=mock.ANY,
                    )
            else:
                # add_scalar() shouldn't be called since is_primary() is False
                summary_writer.add_scalar.assert_not_called()
            summary_writer.add_scalar.reset_mock()

    def test_logged_lr(self):
        # Mock LR scheduler
        class SchedulerMock(ClassyParamScheduler):
            def __call__(self, where):
                return where

        mock_lr_scheduler = SchedulerMock(UpdateInterval.STEP)

        # Mock Logging
        class DummySummaryWriter(object):
            def __init__(self):
                self.scalar_logs = {}

            def add_scalar(self, key, value, global_step=None, walltime=None) -> None:
                self.scalar_logs[key] = self.scalar_logs.get(key, []) + [value]

            def add_histogram(
                self, key, value, global_step=None, walltime=None
            ) -> None:
                return

            def add_text(self, *args, **kwargs):
                pass

            def flush(self):
                return

        config = get_test_mlp_task_config()
        config["num_epochs"] = 3
        config["dataset"]["train"]["batchsize_per_replica"] = 10
        config["dataset"]["test"]["batchsize_per_replica"] = 5
        task = build_task(config)

        writer = DummySummaryWriter()
        hook = TensorboardPlotHook(writer)
        hook.log_period = 1
        task.set_hooks([hook])
        task.set_optimizer_schedulers({"lr": mock_lr_scheduler})

        trainer = LocalTrainer()
        trainer.train(task)

        # We have 20 samples, batch size is 10. Each epoch is done in two steps.
        self.assertEqual(
            writer.scalar_logs["Learning Rate/train"],
            [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6],
        )
