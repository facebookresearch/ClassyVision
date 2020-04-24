#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
import unittest.mock as mock
from itertools import product
from test.generic.config_utils import get_test_mlp_task_config, get_test_task_config
from test.generic.hook_test_utils import HookTestBase

from classy_vision.hooks import ClassyHook, LossLrMeterLoggingHook
from classy_vision.optim.param_scheduler import UpdateInterval
from classy_vision.tasks import ClassyTask, build_task
from classy_vision.trainer import LocalTrainer


class TestLossLrMeterLoggingHook(HookTestBase):
    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        config = {"log_freq": 1}
        invalid_config = copy.deepcopy(config)
        invalid_config["log_freq"] = "this is not an int"

        self.constructor_test_helper(
            config=config,
            hook_type=LossLrMeterLoggingHook,
            hook_registry_name="loss_lr_meter_logging",
            invalid_configs=[invalid_config],
        )

    @mock.patch("classy_vision.hooks.loss_lr_meter_logging_hook.get_rank")
    def test_logging(self, mock_get_rank: mock.MagicMock) -> None:
        """
        Test that the logging happens as expected and the loss and lr values are
        correct.
        """
        rank = 5
        mock_get_rank.return_value = rank

        # set up the task and state
        config = get_test_task_config()
        config["dataset"]["train"]["batchsize_per_replica"] = 2
        config["dataset"]["test"]["batchsize_per_replica"] = 5
        task = build_task(config)
        task.prepare()
        task.on_start()
        task.on_phase_start()

        losses = [1.2, 2.3, 3.4, 4.5]

        task.phase_idx = 0

        for log_freq in [5, None]:
            # create a loss lr meter hook
            loss_lr_meter_hook = LossLrMeterLoggingHook(log_freq=log_freq)

            # check that _log_loss_meters() is called after on_step() every
            # log_freq batches and after on_phase_end()
            # and _log_lr() is called after on_step() every log_freq batches
            # and after on_phase_end()
            with mock.patch.object(loss_lr_meter_hook, "_log_loss_meters") as mock_fn:
                num_batches = 20

                for i in range(num_batches):
                    task.losses = list(range(i))
                    loss_lr_meter_hook.on_step(task)
                    if log_freq is not None and i and i % log_freq == 0:
                        mock_fn.assert_called()
                        mock_fn.reset_mock()
                        continue
                    mock_fn.assert_not_called()

                loss_lr_meter_hook.on_phase_end(task)
                mock_fn.assert_called()

            # test _log_loss_lr_meters()
            task.losses = losses

            with self.assertLogs():
                loss_lr_meter_hook._log_loss_meters(task)

            task.phase_idx += 1

    def test_logged_lr(self):
        # Mock LR scheduler
        def scheduler_mock(where):
            return where

        mock_lr_scheduler = mock.Mock(side_effect=scheduler_mock)
        mock_lr_scheduler.update_interval = UpdateInterval.STEP
        config = get_test_mlp_task_config()
        config["num_epochs"] = 3
        config["dataset"]["train"]["batchsize_per_replica"] = 10
        config["dataset"]["test"]["batchsize_per_replica"] = 5
        task = build_task(config)
        task.optimizer.param_schedulers["lr"] = mock_lr_scheduler
        trainer = LocalTrainer()

        # 2 LR updates per epoch = 6
        lr_order = [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
        lr_list = []

        class LRLoggingHook(ClassyHook):
            on_end = ClassyHook._noop
            on_phase_end = ClassyHook._noop
            on_phase_start = ClassyHook._noop
            on_start = ClassyHook._noop

            def on_step(self, task):
                if task.train:
                    lr_list.append(task.optimizer.parameters.lr)

        hook = LRLoggingHook()
        task.set_hooks([hook])
        trainer.train(task)
        self.assertEqual(lr_list, lr_order)
