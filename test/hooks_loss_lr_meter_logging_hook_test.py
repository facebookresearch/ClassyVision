#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock as mock
from itertools import product
from test.generic.config_utils import get_test_task_config

from classy_vision.hooks import LossLrMeterLoggingHook
from classy_vision.tasks import build_task


class TestLossLrMeterLoggingHook(unittest.TestCase):
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

        losses = [1.2, 2.3, 3.4, 4.5]

        local_variables = {}
        task.phase_idx = 0

        loss_vals = {"train": 1.425, "test": 0.57}

        for log_freq, phase_type in product([5, None], loss_vals):
            task.train = phase_type == "train"

            # create a loss lr meter hook
            loss_lr_meter_hook = LossLrMeterLoggingHook(log_freq=log_freq)

            # check that _log_loss_lr_meters() is called after on_loss() every
            # log_freq batches and after on_phase_end()
            with mock.patch.object(
                loss_lr_meter_hook, "_log_loss_lr_meters"
            ) as mock_fn:
                num_batches = 20

                for i in range(num_batches):
                    task.losses = list(range(i))
                    loss_lr_meter_hook.on_loss(task, local_variables)
                    if log_freq is not None and i and i % log_freq == 0:
                        mock_fn.assert_called_with(task, local_variables)
                        mock_fn.reset_mock()
                        continue
                    mock_fn.assert_not_called()

                loss_lr_meter_hook.on_phase_end(task, local_variables)
                mock_fn.assert_called_with(task, local_variables)

            # test _log_loss_lr_meters()
            task.losses = losses

            with self.assertLogs():
                loss_lr_meter_hook._log_loss_lr_meters(task, local_variables)

            task.phase_idx += 1
