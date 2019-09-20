#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any, Dict, List, Optional

from classy_vision.generic.distributed_util import is_master
from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.state.classy_state import ClassyState


try:
    from tensorboardX import SummaryWriter  # noqa F401

    tbx_available = True
except ImportError:
    tbx_available = False


log = logging.getLogger()


class LossLrTensorboardHook(ClassyHook):
    """
    Hook for writing the losses and learning rates to tensorboard.

    Global steps are counted in terms of the number of samples processed.

    Args:
        tb_writer: Tensorboard SummaryWriter instance
    """

    on_rendezvous = ClassyHook._noop
    on_start = ClassyHook._noop
    on_sample = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, tb_writer) -> None:
        if not tbx_available:
            raise RuntimeError(
                "tensorboardX not installed, cannot use LossLrTensorboardHook"
            )

        self.tb_writer = tb_writer
        self.learning_rates: Optional[List[float]] = None
        self.wall_times: Optional[List[float]] = None
        self.num_steps_global: Optional[List[int]] = None

    def on_phase_start(
        self, state: ClassyState, local_variables: Dict[str, Any]
    ) -> None:
        """
        Initialize losses and learning_rates.
        """
        self.learning_rates = []
        self.wall_times = []
        self.num_steps_global = []

    def on_loss(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Store the observed learning rates.
        """
        if self.learning_rates is None:
            logging.warning("learning_rates is not initialized")
            return

        if not state.train:
            # Only need to log the average loss during the test phase
            return

        learning_rate_val = state.optimizer.optimizer_config["lr"]

        self.learning_rates.append(learning_rate_val)
        self.wall_times.append(time.time())
        self.num_steps_global.append(state.num_updates)

    def on_phase_end(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Add the losses and learning rates to tensorboard.
        """
        if self.learning_rates is None:
            logging.warning("learning_rates is not initialized")
            return

        batches = len(state.losses)
        if batches == 0 or not is_master():
            return

        phase_type = state.phase_type
        phase_type_idx = state.train_phase_idx if state.train else state.eval_phase_idx

        phase_type = state.phase_type
        loss_key = f"{phase_type}_loss"
        learning_rate_key = f"{phase_type}_learning_rate_updates"

        if state.train:
            for loss, learning_rate, global_step, wall_time in zip(
                state.losses,
                self.learning_rates,
                self.num_steps_global,
                self.wall_times,
            ):
                loss /= state.get_batchsize_per_replica()
                self.tb_writer.add_scalar(
                    loss_key, loss, global_step=global_step, walltime=wall_time
                )
                self.tb_writer.add_scalar(
                    learning_rate_key,
                    learning_rate,
                    global_step=global_step,
                    walltime=wall_time,
                )

        loss_avg = sum(state.losses) / (batches * state.get_batchsize_per_replica())

        loss_key = "avg_{phase_type}_loss".format(phase_type=state.phase_type)
        self.tb_writer.add_scalar(loss_key, loss_avg, global_step=phase_type_idx)
