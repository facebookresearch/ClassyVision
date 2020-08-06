#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from itertools import accumulate
from typing import Any, Dict, List, Optional, Tuple

import torch
from classy_vision.generic.distributed_util import all_reduce_max, is_primary
from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook


try:
    from torch.utils.tensorboard import SummaryWriter  # noqa F401

    tb_available = True
except ImportError:
    tb_available = False


log = logging.getLogger()


@register_hook("tensorboard_plot")
class TensorboardPlotHook(ClassyHook):
    """
    Hook for writing the losses, learning rates and meters to `tensorboard <https
    ://www.tensorflow.org/tensorboard`>_.

    Global steps are counted in terms of the number of samples processed.
    """

    on_end = ClassyHook._noop

    def __init__(self, tb_writer, log_period: int = 10) -> None:
        """The constructor method of TensorboardPlotHook.

        Args:
            tb_writer: `Tensorboard SummaryWriter <https://tensorboardx.
            readthedocs.io/en/latest/tensorboard.html#tensorboardX.
            SummaryWriter>`_ instance or None (only on non-master replicas)
        """
        super().__init__()
        if not tb_available:
            raise ModuleNotFoundError(
                "tensorboard not installed, cannot use TensorboardPlotHook"
            )
        if not isinstance(log_period, int):
            raise TypeError("log_period must be an int")
        self.tb_writer = tb_writer
        self.learning_rates: Optional[List[float]] = None
        self.wall_times: Optional[List[float]] = None
        self.sample_fetch_times: Optional[List[float]] = None
        self.log_period = log_period
        # need to maintain the step count at the end of every phase
        # and the cumulative sample fetch time for checkpointing
        self.state.step_count = {"train": 0, "test": 0}
        self.state.cum_sample_fetch_time = {"train": 0, "test": 0}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TensorboardPlotHook":
        """The config is expected to include the key
        "summary_writer" with arguments which correspond
        to those listed at <https://tensorboardx.
        readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter>:

        """
        tb_writer = SummaryWriter(**config["summary_writer"])
        log_period = config.get("log_period", 10)
        return cls(tb_writer=tb_writer, log_period=log_period)

    def on_start(self, task) -> None:
        if is_primary():
            self.tb_writer.add_text("Task", f"{task}")

    def on_phase_start(self, task) -> None:
        """Initialize losses and learning_rates."""
        self.learning_rates = []
        self.wall_times = []
        self.sample_fetch_times = []

        if not is_primary():
            return

        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

        # log the parameters before training starts
        if task.train and task.train_phase_idx == 0:
            for name, parameter in task.base_model.named_parameters():
                self.tb_writer.add_histogram(
                    f"Parameters/{name}", parameter, global_step=-1
                )

    def on_step(self, task) -> None:
        """Store the observed learning rates."""
        self.state.step_count[task.phase_type] += 1
        self.wall_times.append(time.time())
        if "sample_fetch_time" in task.last_batch.step_data:
            self.sample_fetch_times.append(
                task.last_batch.step_data["sample_fetch_time"]
            )
        if task.train:
            self.learning_rates.append(task.optimizer.options_view.lr)

    def _get_cum_sample_fetch_times(self, phase_type) -> Tuple[List[float], ...]:
        if not self.sample_fetch_times:
            return None

        sample_fetch_times = torch.Tensor(self.sample_fetch_times)
        max_sample_fetch_times = all_reduce_max(sample_fetch_times).tolist()
        cum_sample_fetch_times = list(
            accumulate(
                [self.state.cum_sample_fetch_time[phase_type]] + max_sample_fetch_times
            )
        )[1:]
        self.state.cum_sample_fetch_time[phase_type] = cum_sample_fetch_times[-1]
        return cum_sample_fetch_times

    def on_phase_end(self, task) -> None:
        """Add the losses and learning rates to tensorboard."""
        if self.learning_rates is None:
            logging.warning("learning_rates is not initialized")
            return

        phase_type = task.phase_type
        cum_sample_fetch_times = self._get_cum_sample_fetch_times(phase_type)

        batches = len(task.losses)
        if batches == 0 or not is_primary():
            return

        phase_type_idx = task.train_phase_idx if task.train else task.eval_phase_idx

        logging.info(f"Plotting to Tensorboard for {phase_type} phase {phase_type_idx}")

        for i in range(0, len(self.wall_times), self.log_period):
            global_step = (
                i + self.state.step_count[phase_type] - len(self.wall_times) + 1
            )
            if cum_sample_fetch_times:
                self.tb_writer.add_scalar(
                    f"Speed/{phase_type}/cumulative_sample_fetch_time",
                    cum_sample_fetch_times[i],
                    global_step=global_step,
                    walltime=self.wall_times[i],
                )
            if task.train:
                self.tb_writer.add_scalar(
                    "Learning Rate/train",
                    self.learning_rates[i],
                    global_step=global_step,
                    walltime=self.wall_times[i],
                )

        if task.train:
            for name, parameter in task.base_model.named_parameters():
                self.tb_writer.add_histogram(
                    f"Parameters/{name}", parameter, global_step=phase_type_idx
                )

        if torch.cuda.is_available() and task.train:
            self.tb_writer.add_scalar(
                "Memory/peak_allocated",
                torch.cuda.max_memory_allocated(),
                global_step=phase_type_idx,
            )

        loss_avg = sum(task.losses) / (batches * task.get_batchsize_per_replica())

        loss_key = "Losses/{phase_type}".format(phase_type=task.phase_type)
        self.tb_writer.add_scalar(loss_key, loss_avg, global_step=phase_type_idx)

        # plot meters which return a dict
        for meter in task.meters:
            if not isinstance(meter.value, dict):
                log.warn(f"Skipping meter {meter.name} with value: {meter.value}")
                continue
            for name, value in meter.value.items():
                if isinstance(value, float):
                    meter_key = f"Meters/{phase_type}/{meter.name}/{name}"
                    self.tb_writer.add_scalar(
                        meter_key, value, global_step=phase_type_idx
                    )
                else:
                    log.warn(
                        f"Skipping meter name {meter.name}/{name} with value: {value}"
                    )
                    continue

        if hasattr(task, "perf_log"):
            for perf in task.perf_log:
                phase_idx = perf["phase_idx"]
                tag = perf["tag"]
                for metric_name, metric_value in perf.items():
                    if metric_name in ["phase_idx", "tag"]:
                        continue

                    self.tb_writer.add_scalar(
                        f"Speed/{tag}/{metric_name}",
                        metric_value,
                        global_step=phase_idx,
                    )

        # flush so that the plots aren't lost if training crashes soon after
        self.tb_writer.flush()
        logging.info("Done plotting to Tensorboard")
