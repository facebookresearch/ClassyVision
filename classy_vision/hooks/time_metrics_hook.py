#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any, Dict, Optional

from classy_vision import tasks
from classy_vision.generic.distributed_util import get_rank
from classy_vision.generic.perf_stats import PerfStats
from classy_vision.hooks.classy_hook import ClassyHook


class TimeMetricsHook(ClassyHook):
    """
    Computes and prints performance metrics. Logs at the end of a phase.
    """

    on_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, log_freq: Optional[int] = None) -> None:
        """
        Args:
            log_freq: if specified, logs every log_freq batches also.
        """
        super().__init__()
        self.log_freq: Optional[int] = log_freq
        self.start_time: Optional[float] = None

    def on_phase_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Initialize start time and reset perf stats
        """
        self.start_time = time.time()
        local_variables["perf_stats"] = PerfStats()

    def on_step(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Log metrics every log_freq batches, if log_freq is not None.
        """
        if self.log_freq is None:
            return
        batches = len(task.losses)
        if batches and batches % self.log_freq == 0:
            self._log_performance_metrics(task, local_variables)

    def on_phase_end(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Log metrics at the end of a phase if log_freq is None.
        """
        batches = len(task.losses)
        if batches:
            self._log_performance_metrics(task, local_variables)

    def _log_performance_metrics(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Compute and log performance metrics.
        """
        phase_type = task.phase_type
        batches = len(task.losses)

        if self.start_time is None:
            logging.warning("start_time not initialized")
        else:
            # Average batch time calculation
            total_batch_time = time.time() - self.start_time
            average_batch_time = total_batch_time / batches
            logging.info(
                "Average %s batch time (ms) for %d batches: %d"
                % (phase_type, batches, 1000.0 * average_batch_time)
            )

        # Train step time breakdown
        if local_variables.get("perf_stats") is None:
            logging.warning('"perf_stats" not set in local_variables')
        elif task.train:
            logging.info(
                "Train step time breakdown (rank {}):\n{}".format(
                    get_rank(), local_variables["perf_stats"].report_str()
                )
            )
