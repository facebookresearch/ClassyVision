#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any, Dict, List, Optional
from functools import partial
from collections import defaultdict

from classy_vision.generic.distributed_util import is_master
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

    on_start = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, tb_writer, log_period: int = 10) -> None:
        """The constructor method of TensorboardPlotHook.

        Args:
            tb_writer: `Tensorboard SummaryWriter <https://tensorboardx.
            readthedocs.io/en/latest/tensorboard.html#tensorboardX.
            SummaryWriter>`_ instance
        """
        super().__init__()
        if not tb_available:
            raise RuntimeError(
                "tensorboard not installed, cannot use TensorboardPlotHook"
            )
        assert isinstance(log_period, int), "log_period must be an int"

        self.tb_writer = tb_writer
        self.learning_rates: Optional[List[float]] = None
        self.wall_times: Optional[List[float]] = None
        self.num_updates: Optional[List[int]] = None
        self.log_period = log_period

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

    def fhook(self, layer_id, mod, inp, output):
        self.activation_mean[layer_id] = output.data.mean()
        self.activation_std[layer_id] = output.data.std()

    def on_start(self, task):
        for layer_id, module in enumerate(task.base_model.classy_model):
            module.register_forward_hook(partial(self.fhook, layer_id))

    def on_phase_start(self, task) -> None:
        """Initialize losses and learning_rates."""
        self.learning_rates = []
        self.wall_times = []
        self.num_updates = []
        self.step_idx = 0
        self.activation_mean = defaultdict(list)
        self.activation_std = defaultdict(list)

        if not is_master():
            return

        # log the parameters before training starts
        if task.train and task.train_phase_idx == 0:
            for name, parameter in task.base_model.named_parameters():
                self.tb_writer.add_histogram(
                    f"Parameters/{name}", parameter, global_step=-1
                )

    def on_step(self, task) -> None:
        """Store the observed learning rates."""
        if self.learning_rates is None:
            logging.warning("learning_rates is not initialized")
            return

        if not task.train:
            # Only need to log the average loss during the test phase
            return

        if self.step_idx % self.log_period == 0:
            learning_rate_val = task.optimizer.parameters.lr

            self.learning_rates.append(learning_rate_val)
            self.wall_times.append(time.time())
            self.num_updates.append(task.num_updates)

            self.tb_writer.add_scalars("Activation/mean", {f'layer {layer_id}': mean for layer_id, mean in self.activation_mean.items()}, global_step=task.num_updates)
            self.tb_writer.add_scalars("Activation/std", {f'layer {layer_id}': mean for layer_id, mean in self.activation_std.items()}, global_step=task.num_updates)

        self.step_idx += 1

    def on_phase_end(self, task) -> None:
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
        learning_rate_key = f"Learning_Rate/{phase_type}"

        if task.train:
            for learning_rate, global_step, wall_time in zip(
                self.learning_rates, self.num_updates, self.wall_times
            ):
                self.tb_writer.add_scalar(
                    learning_rate_key,
                    learning_rate,
                    global_step=global_step,
                    walltime=wall_time,
                )
            for name, parameter in task.base_model.named_parameters():
                self.tb_writer.add_histogram(
                    f"Parameters/{name}", parameter, global_step=phase_type_idx
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
        logging.info(f"Done plotting to Tensorboard")
