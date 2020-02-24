#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import unittest
import unittest.mock as mock
from itertools import product
from test.generic.config_utils import get_test_classy_task

from classy_vision.generic.perf_stats import PerfStats
from classy_vision.hooks import TimeMetricsHook


class TestTimeMetricsHook(unittest.TestCase):
    @mock.patch("time.time")
    @mock.patch("classy_vision.hooks.time_metrics_hook.PerfStats.report_str")
    @mock.patch("classy_vision.hooks.time_metrics_hook.get_rank")
    def test_time_metrics(
        self,
        mock_get_rank: mock.MagicMock,
        mock_report_str: mock.MagicMock,
        mock_time: mock.MagicMock,
    ) -> None:
        """
        Tests that the progress bar is created, updated and destroyed correctly.
        """
        rank = 5
        mock_get_rank.return_value = rank

        mock_report_str.return_value = ""
        local_variables = {}

        for log_freq, train in product([5, None], [True, False]):
            # create a time metrics hook
            time_metrics_hook = TimeMetricsHook(log_freq=log_freq)

            phase_type = "train" if train else "test"

            task = get_test_classy_task()
            task.prepare()
            task.train = train

            # on_phase_start() should set the start time and perf_stats
            start_time = 1.2
            mock_time.return_value = start_time
            time_metrics_hook.on_phase_start(task, local_variables)
            self.assertEqual(time_metrics_hook.start_time, start_time)
            self.assertTrue(isinstance(local_variables.get("perf_stats"), PerfStats))

            # test that the code doesn't raise an exception if losses is empty
            try:
                time_metrics_hook.on_phase_end(task, local_variables)
            except Exception as e:
                self.fail("Received Exception when losses is []: {}".format(e))

            # check that _log_performance_metrics() is called after on_step()
            # every log_freq batches and after on_phase_end()
            with mock.patch.object(
                time_metrics_hook, "_log_performance_metrics"
            ) as mock_fn:
                num_batches = 20

                for i in range(num_batches):
                    task.losses = list(range(i))
                    time_metrics_hook.on_step(task, local_variables)
                    if log_freq is not None and i and i % log_freq == 0:
                        mock_fn.assert_called_with(task, local_variables)
                        mock_fn.reset_mock()
                        continue
                    mock_fn.assert_not_called()

                time_metrics_hook.on_phase_end(task, local_variables)
                mock_fn.assert_called_with(task, local_variables)

            task.losses = [0.23, 0.45, 0.34, 0.67]

            end_time = 10.4
            avg_batch_time_ms = 2.3 * 1000
            mock_time.return_value = end_time

            # test _log_performance_metrics()
            with self.assertLogs() as log_watcher:
                time_metrics_hook._log_performance_metrics(task, local_variables)

            # there should 2 be info logs for train and 1 for test
            self.assertEqual(len(log_watcher.output), 2 if train else 1)
            self.assertTrue(
                all(
                    log_record.levelno == logging.INFO
                    for log_record in log_watcher.records
                )
            )
            match = re.search(
                (
                    r"Average {} batch time \(ms\) for {} batches: "
                    r"(?P<avg_batch_time>[-+]?\d*\.\d+|\d+)"
                ).format(phase_type, len(task.losses)),
                log_watcher.output[0],
            )
            self.assertIsNotNone(match)
            self.assertAlmostEqual(
                avg_batch_time_ms, float(match.group("avg_batch_time")), places=4
            )
            if train:
                self.assertIn(
                    f"Train step time breakdown (rank {rank})", log_watcher.output[1]
                )

            # if on_phase_start() is not called, 2 warnings should be logged
            # create a new time metrics hook
            local_variables = {}
            time_metrics_hook_new = TimeMetricsHook()

            with self.assertLogs() as log_watcher:
                time_metrics_hook_new.on_phase_end(task, local_variables)

            self.assertEqual(len(log_watcher.output), 2)
            self.assertTrue(
                all(
                    log_record.levelno == logging.WARN
                    for log_record in log_watcher.records
                )
            )
