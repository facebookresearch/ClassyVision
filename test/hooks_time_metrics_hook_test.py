#!/usr/bin/env python3

import logging
import re
import unittest
import unittest.mock as mock
from itertools import product
from test.generic.config_utils import get_test_classy_task

from classy_vision.generic.perf_stats import PerfStats
from classy_vision.hooks.time_metrics_hook import TimeMetricsHook


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
            state = task.build_initial_state()
            state.train = train

            # on_phase_start() should set the start time and perf_stats
            start_time = 1.2
            mock_time.return_value = start_time
            time_metrics_hook.on_phase_start(state, local_variables)
            self.assertEqual(time_metrics_hook.start_time, start_time)
            self.assertTrue(isinstance(local_variables.get("perf_stats"), PerfStats))

            # test that the code doesn't raise an exception if losses is empty
            try:
                time_metrics_hook.on_phase_end(state, local_variables)
            except Exception as e:
                self.fail("Received Exception when losses is []: {}".format(e))

            # check that _log_performance_metrics() is called after on_loss() every
            # log_freq batches and after on_phase_end()
            with mock.patch.object(
                time_metrics_hook, "_log_performance_metrics"
            ) as mock_fn:
                num_batches = 20

                for i in range(num_batches):
                    state.losses = list(range(i))
                    time_metrics_hook.on_loss(state, local_variables)
                    if log_freq is not None and i and i % log_freq == 0:
                        mock_fn.assert_called_with(state, local_variables)
                        mock_fn.reset_mock()
                        continue
                    mock_fn.assert_not_called()

                time_metrics_hook.on_phase_end(state, local_variables)
                mock_fn.assert_called_with(state, local_variables)

            state.losses = [0.23, 0.45, 0.34, 0.67]

            end_time = 10.4
            avg_batch_time_ms = 2.3 * 1000
            mock_time.return_value = end_time

            # test _log_performance_metrics()
            with self.assertLogs() as log_watcher:
                time_metrics_hook._log_performance_metrics(state, local_variables)

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
                ).format(phase_type, len(state.losses)),
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
                time_metrics_hook_new.on_phase_end(state, local_variables)

            self.assertEqual(len(log_watcher.output), 2)
            self.assertTrue(
                all(
                    log_record.levelno == logging.WARN
                    for log_record in log_watcher.records
                )
            )
