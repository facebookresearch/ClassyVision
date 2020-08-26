#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from classy_vision.generic.distributed_util import get_rank
from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook


@register_hook("loss_lr_meter_logging")
class LossLrMeterLoggingHook(ClassyHook):
    """
    Logs the loss, optimizer LR, and meters. Logs at the end of a phase.
    """

    on_phase_start = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, log_freq: Optional[int] = None) -> None:
        """The constructor method of LossLrMeterLoggingHook.

        Args:
            log_freq: if specified, also logs every ``log_freq`` batches.

        """
        super().__init__()
        assert log_freq is None or isinstance(
            log_freq, int
        ), "log_freq must be an int or None"
        self.log_freq: Optional[int] = log_freq

    def on_start(self, task) -> None:
        logging.info(f"Starting training. Task: {task}")

    def on_phase_end(self, task) -> None:
        """
        Log the loss, optimizer LR, and meters for the phase.
        """
        batches = len(task.losses)
        if batches:
            # Most trainers will sync meters on phase end, however we
            # do not explicitly state this since it is possible for a
            # trainer to implement an unsynced end of phase meter or
            # for meters to not provide a sync function.
            self._log_loss_lr_meters(task, prefix="Synced meters: ", log_batches=True)

    def on_step(self, task) -> None:
        """
        Log the LR every log_freq batches, if log_freq is not None.
        """
        if self.log_freq is None or not task.train:
            return
        batches = len(task.losses)
        if batches and batches % self.log_freq == 0:
            self._log_loss_lr_meters(task, prefix="Approximate meters: ")

    def _log_loss_lr_meters(self, task, prefix="", log_batches=False) -> None:
        """
        Compute and log the loss, lr, and meters.
        """

        phase_type = task.phase_type
        phase_type_idx = task.train_phase_idx if task.train else task.eval_phase_idx
        batches = len(task.losses)

        # Loss for the phase
        loss = sum(task.losses) / (batches * task.get_batchsize_per_replica())
        phase_pct = batches / task.num_batches_per_phase
        msg = (
            f"{prefix}[{get_rank()}] {phase_type} phase {phase_type_idx} "
            f"({phase_pct*100:.2f}% done), loss: {loss:.4f}, meters: {task.meters}"
        )
        if task.train:
            msg += f", lr: {task.optimizer.options_view.lr:.4f}"
        if phase_type == "test" and hasattr(task, "ema"):
            msg += f", ema: {task.ema}"
        if log_batches:
            msg += f", processed batches: {batches}"

        logging.info(msg)
