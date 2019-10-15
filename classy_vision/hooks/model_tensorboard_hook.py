#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

from classy_vision.generic.distributed_util import is_master
from classy_vision.generic.visualize import plot_model
from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.state.classy_state import ClassyState


try:
    from tensorboardX import SummaryWriter  # noqa F401

    tbx_available = True
except ImportError:
    tbx_available = False


class ModelTensorboardHook(ClassyHook):
    """
    Shows the model graph in TensorBoard.
    """

    on_rendezvous = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_sample = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, tb_writer) -> None:
        super().__init__()
        if not tbx_available:
            raise RuntimeError(
                "tensorboardX not installed, cannot use ModelTensorboardHook"
            )

        self.tb_writer = tb_writer

    def on_start(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Plot the model on Tensorboard.
        """
        # Show model in tensorboard:
        logging.info("Showing model graph in TensorBoard...")

        if is_master():
            try:
                plot_model(
                    state.base_model,
                    size=state.base_model.input_shape,
                    input_key=state.base_model.input_key
                    if hasattr(state.base_model, "input_key")
                    else None,
                    writer=self.tb_writer,
                )
            except Exception:
                logging.warn(
                    "Unable to plot model to tensorboard. Exception: ", exc_info=True
                )
