#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional

from classy_vision import tasks
from classy_vision.generic.distributed_util import get_rank
from classy_vision.hooks.classy_hook import ClassyHook


class LossLrMeterLoggingHook(ClassyHook):
    """
    Logs the loss, optimizer LR, and meters. Logs at the end of a phase.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, log_freq: Optional[int] = None) -> None:
        """The constructor method of LossLrMeterLoggingHook.

        Args:
            log_freq: if specified, also logs every ``log_freq`` batches.

        """
        super().__init__()
        self.log_freq: Optional[int] = log_freq

    def on_phase_end(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Log the loss, optimizer LR, and meters for the phase.
        """
        batches = len(task.losses)
        if batches:
            # Most trainers will sync meters on phase end, however we
            # do not explicitly state this since it is possible for a
            # trainer to implement an unsynced end of phase meter or
            # for meters to not provide a sync function.
            logging.info("End of phase metric values:")
            self._log_loss_meters(task, local_variables)
            if task.train:
                self._log_lr(task, local_variables)

    def on_step(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Log the LR every log_freq batches, if log_freq is not None.
        """
        if self.log_freq is None or not task.train:
            return
        batches = len(task.losses)
        if batches and batches % self.log_freq == 0:
            self._log_lr(task, local_variables)
            logging.info("Local unsynced metric values:")
            self._log_loss_meters(task, local_variables)

    def _log_lr(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Compute and log the optimizer LR.
        """
        optimizer_lr = task.optimizer.parameters.lr
        logging.info("Learning Rate: {}\n".format(optimizer_lr))

    def _log_loss_meters(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Compute and log the loss and meters.
        """

        phase_type = task.phase_type
        phase_type_idx = task.train_phase_idx if task.train else task.eval_phase_idx
        batches = len(task.losses)

        # Loss for the phase
        loss = sum(task.losses) / (batches * task.get_batchsize_per_replica())

        log_strs = [
            "Rank: {}, {} phase: {}, processed batches: {}".format(
                get_rank(), phase_type, phase_type_idx, batches
            ),
            "{} loss: {}".format(phase_type, loss),
            "Meters:",
        ]
        for meter in task.meters:
            log_strs.append("{}".format(meter))
        logging.info("\n".join(log_strs))
