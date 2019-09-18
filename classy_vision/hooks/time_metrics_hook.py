#!/usr/bin/env python3

import logging
import time
from typing import Any, Dict, Optional

from classy_vision.generic.distributed_util import get_rank
from classy_vision.generic.perf_stats import PerfStats
from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.state.classy_state import ClassyState


class TimeMetricsHook(ClassyHook):
    """
    Computes and prints performance metrics. Logs at the end of a phase.

    if log_freq is specified, logs every log_freq batches also.
    """

    on_rendezvous = ClassyHook._noop
    on_start = ClassyHook._noop
    on_sample = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, log_freq: Optional[int] = None) -> None:
        self.log_freq: Optional[int] = log_freq
        self.start_time: Optional[float] = None

    def on_phase_start(
        self, state: ClassyState, local_variables: Dict[str, Any]
    ) -> None:
        """
        Initialize start time and reset perf stats
        """
        self.start_time = time.time()
        local_variables["perf_stats"] = PerfStats()

    def on_loss(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Log metrics every log_freq batches, if log_freq is not None.
        """
        if self.log_freq is None:
            return
        batches = len(state.losses)
        if batches and batches % self.log_freq == 0:
            self._log_performance_metrics(state, local_variables)

    def on_phase_end(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Log metrics at the end of a phase if log_freq is None.
        """
        batches = len(state.losses)
        if batches:
            self._log_performance_metrics(state, local_variables)

    def _log_performance_metrics(
        self, state: ClassyState, local_variables: Dict[str, Any]
    ) -> None:
        """
        Compute and log performance metrics.
        """
        phase_type = state.phase_type
        batches = len(state.losses)

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
        elif state.train:
            logging.info(
                "Train step time breakdown (rank {}):\n{}".format(
                    get_rank(), local_variables["perf_stats"].report_str()
                )
            )
