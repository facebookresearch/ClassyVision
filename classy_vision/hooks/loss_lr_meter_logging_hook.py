#!/usr/bin/env python3

import logging
from math import floor
from typing import Any, Dict, Optional

from classy_vision.generic.distributed_util import get_rank
from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.state.classy_state import ClassyState


class LossLrMeterLoggingHook(ClassyHook):
    """
    Logs the loss, optimizer LR, and meters. Logs at the end of a phase.

    if log_freq is specified, logs every log_freq batches also.
    """

    on_rendezvous = ClassyHook._noop
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_sample = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, log_freq: Optional[int] = None) -> None:
        self.log_freq: Optional[int] = log_freq

    def on_loss(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Log metrics every log_freq batches, if log_freq is not None.
        """
        if self.log_freq is None:
            return
        batches = len(state.losses)
        if batches and batches % self.log_freq == 0:
            self._log_loss_lr_meters(state, local_variables)

    def on_phase_end(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Log the loss, optimizer LR, and meters for the phase.
        """
        batches = len(state.losses)
        if batches:
            self._log_loss_lr_meters(state, local_variables)

    def _log_loss_lr_meters(
        self, state: ClassyState, local_variables: Dict[str, Any]
    ) -> None:
        """
        Compute and log the loss, optimizer LR, and meters.
        """
        phase_type = state.phase_type
        phase_type_idx = state.train_phase_idx if state.train else state.eval_phase_idx
        batches = len(state.losses)

        # Loss for the phase
        loss = sum(state.losses) / (batches * state.get_batchsize_per_replica())

        # Optimizer LR for the phase
        optimizer_lr = state.optimizer.optimizer_config["lr"]

        log_strs = [
            "Rank: {}, {} phase: {}, processed batches: {}".format(
                get_rank(), phase_type, phase_type_idx, batches
            ),
            "{} loss: {}, LR rate: {}".format(phase_type, loss, optimizer_lr),
            "Meters:",
        ]
        for meter in state.meters:
            log_strs.append("{}".format(meter))
        logging.info("\n".join(log_strs))
