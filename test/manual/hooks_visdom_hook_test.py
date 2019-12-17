#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
import unittest.mock as mock
from itertools import product
from test.generic.config_utils import get_test_task_config

from classy_vision.hooks import VisdomHook
from classy_vision.tasks import build_task
from visdom import Visdom


class TestVisdomHook(unittest.TestCase):
    @mock.patch("classy_vision.hooks.visdom_hook.is_master")
    @mock.patch("classy_vision.hooks.visdom_hook.Visdom", autospec=True)
    def test_visdom(
        self, mock_visdom_cls: mock.MagicMock, mock_is_master: mock.MagicMock
    ) -> None:
        """
        Tests that visdom is populated with plots.
        """
        mock_visdom = mock.create_autospec(Visdom, instance=True)
        mock_visdom_cls.return_value = mock_visdom

        local_variables = {}

        # set up the task and state
        config = get_test_task_config()
        config["dataset"]["train"]["batchsize_per_replica"] = 2
        config["dataset"]["test"]["batchsize_per_replica"] = 5
        task = build_task(config)
        task.prepare()

        losses = [1.2, 2.3, 1.23, 2.33]
        loss_vals = {"train": 0.8825, "test": 0.353}

        task.losses = losses

        visdom_server = "localhost"
        visdom_port = 8097

        for master, visdom_conn in product([False, True], [False, True]):
            mock_is_master.return_value = master
            mock_visdom.check_connection.return_value = visdom_conn

            # create a visdom hook
            visdom_hook = VisdomHook(visdom_server, visdom_port)

            mock_visdom_cls.assert_called_once()
            mock_visdom_cls.reset_mock()

            counts = {"train": 0, "test": 0}
            count = 0

            for phase_idx in range(10):
                train = phase_idx % 2 == 0
                task.train = train
                phase_type = "train" if train else "test"

                counts[phase_type] += 1
                count += 1

                # test that the metrics don't change if losses is empty and that
                # visdom.line() is not called
                task.losses = []
                original_metrics = copy.deepcopy(visdom_hook.metrics)
                visdom_hook.on_phase_end(task, local_variables)
                self.assertDictEqual(original_metrics, visdom_hook.metrics)
                mock_visdom.line.assert_not_called()

                # test that the metrics are updated correctly when losses
                # is non empty
                task.losses = [loss * count for loss in losses]
                visdom_hook.on_phase_end(task, local_variables)

                # every meter should be present and should have the correct length
                for meter in task.meters:
                    for key in meter.value:
                        key = phase_type + "_" + meter.name + "_" + key
                        self.assertTrue(
                            key in visdom_hook.metrics
                            and type(visdom_hook.metrics[key]) == list
                            and len(visdom_hook.metrics[key]) == counts[phase_type]
                        )

                # the loss metric should be calculated correctly
                loss_key = phase_type + "_loss"
                self.assertTrue(
                    loss_key in visdom_hook.metrics
                    and type(visdom_hook.metrics[loss_key]) == list
                    and len(visdom_hook.metrics[loss_key]) == counts[phase_type]
                )
                self.assertAlmostEqual(
                    visdom_hook.metrics[loss_key][-1],
                    loss_vals[phase_type] * count,
                    places=4,
                )

                # the lr metric should be correct
                lr_key = phase_type + "_learning_rate"
                self.assertTrue(
                    lr_key in visdom_hook.metrics
                    and type(visdom_hook.metrics[lr_key]) == list
                    and len(visdom_hook.metrics[lr_key]) == counts[phase_type]
                )
                self.assertAlmostEqual(
                    visdom_hook.metrics[lr_key][-1],
                    task.optimizer.parameters.lr,
                    places=4,
                )

                if master and not train and visdom_conn:
                    # visdom.line() should be called once
                    mock_visdom.line.assert_called_once()
                    mock_visdom.line.reset_mock()
                else:
                    # visdom.line() should not be called
                    mock_visdom.line.assert_not_called()
