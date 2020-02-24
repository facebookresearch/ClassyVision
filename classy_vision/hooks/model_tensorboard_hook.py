#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

from classy_vision import tasks
from classy_vision.generic.distributed_util import is_master
from classy_vision.generic.visualize import plot_model
from classy_vision.hooks.classy_hook import ClassyHook


try:
    from tensorboardX import SummaryWriter  # noqa F401

    tbx_available = True
except ImportError:
    tbx_available = False


class ModelTensorboardHook(ClassyHook):
    """
    Shows the model graph in `TensorBoard <https
    ://www.tensorflow.org/tensorboard`>_.
    """

    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, tb_writer) -> None:
        """The constructor method of ModelTensorboardHook.

        Args:
            tb_writer: `Tensorboard SummaryWriter <https://tensorboardx.
            readthedocs.io/en/latest/tensorboard.html#tensorboardX.
            SummaryWriter>`_ instance

        """
        super().__init__()
        if not tbx_available:
            raise RuntimeError(
                "tensorboardX not installed, cannot use ModelTensorboardHook"
            )

        self.tb_writer = tb_writer

    def on_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Plot the model on Tensorboard.
        """
        if is_master():
            try:
                # Show model in tensorboard:
                logging.info("Showing model graph in TensorBoard...")

                plot_model(
                    task.base_model,
                    size=task.base_model.input_shape,
                    input_key=task.base_model.input_key
                    if hasattr(task.base_model, "input_key")
                    else None,
                    writer=self.tb_writer,
                )
            except Exception:
                logging.warn(
                    "Unable to plot model to tensorboard. Exception: ", exc_info=True
                )
