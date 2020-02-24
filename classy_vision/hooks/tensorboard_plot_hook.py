#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any, Dict, List, Optional

from classy_vision import tasks
from classy_vision.generic.distributed_util import is_master
from classy_vision.hooks.classy_hook import ClassyHook


try:
    from tensorboardX import SummaryWriter  # noqa F401

    tbx_available = True
except ImportError:
    tbx_available = False


log = logging.getLogger()


class TensorboardPlotHook(ClassyHook):
    """
    Hook for writing the losses, learning rates and meters to `tensorboard <https
    ://www.tensorflow.org/tensorboard`>_.

    Global steps are counted in terms of the number of samples processed.
    """

    on_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, tb_writer) -> None:
        """The constructor method of TensorboardPlotHook.

        Args:
            tb_writer: `Tensorboard SummaryWriter <https://tensorboardx.
            readthedocs.io/en/latest/tensorboard.html#tensorboardX.
            SummaryWriter>`_ instance
        """
        super().__init__()
        if not tbx_available:
            raise RuntimeError(
                "tensorboardX not installed, cannot use TensorboardPlotHook"
            )

        self.tb_writer = tb_writer
        self.learning_rates: Optional[List[float]] = None
        self.wall_times: Optional[List[float]] = None
        self.num_steps_global: Optional[List[int]] = None

    def on_phase_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Initialize losses and learning_rates."""
        self.learning_rates = []
        self.wall_times = []
        self.num_steps_global = []

    def on_step(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Store the observed learning rates."""
        if self.learning_rates is None:
            logging.warning("learning_rates is not initialized")
            return

        if not task.train:
            # Only need to log the average loss during the test phase
            return

        learning_rate_val = task.optimizer.parameters.lr

        self.learning_rates.append(learning_rate_val)
        self.wall_times.append(time.time())
        self.num_steps_global.append(task.num_updates)

    def on_phase_end(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Add the losses and learning rates to tensorboard."""
        if self.learning_rates is None:
            logging.warning("learning_rates is not initialized")
            return

        batches = len(task.losses)
        if batches == 0 or not is_master():
            return

        phase_type = task.phase_type
        phase_type_idx = task.train_phase_idx if task.train else task.eval_phase_idx

        logging.info(f"Plotting to Tensorboard for {phase_type} phase {phase_type_idx}")

        phase_type = task.phase_type
        loss_key = f"{phase_type}_loss"
        learning_rate_key = f"{phase_type}_learning_rate_updates"

        if task.train:
            for loss, learning_rate, global_step, wall_time in zip(
                task.losses, self.learning_rates, self.num_steps_global, self.wall_times
            ):
                loss /= task.get_batchsize_per_replica()
                self.tb_writer.add_scalar(
                    loss_key, loss, global_step=global_step, walltime=wall_time
                )
                self.tb_writer.add_scalar(
                    learning_rate_key,
                    learning_rate,
                    global_step=global_step,
                    walltime=wall_time,
                )

        loss_avg = sum(task.losses) / (batches * task.get_batchsize_per_replica())

        loss_key = "avg_{phase_type}_loss".format(phase_type=task.phase_type)
        self.tb_writer.add_scalar(loss_key, loss_avg, global_step=phase_type_idx)

        # plot meters which return a dict
        for meter in task.meters:
            if not isinstance(meter.value, dict):
                log.warn(f"Skipping meter {meter.name} with value: {meter.value}")
                continue
            for name, value in meter.value.items():
                if isinstance(value, float):
                    meter_key = f"{phase_type}_{meter.name}_{name}"
                    self.tb_writer.add_scalar(
                        meter_key, value, global_step=phase_type_idx
                    )
                else:
                    log.warn(
                        f"Skipping meter name {meter.name}_{name} with value: {value}"
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
        logging.info(f"Done plotting to Tensorboard")
