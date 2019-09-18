#!/usr/bin/env python3

import logging
from typing import Any, Dict

from classy_vision.generic.distributed_util import is_master
from classy_vision.generic.visualize import plot_model
from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.state.classy_state import ClassyState


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

    def __init__(self, tensorboard_dir: str = "/tmp/tensorboard") -> None:
        self.tensorboard_dir: str = tensorboard_dir

    def on_start(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Plot the model on Tensorboard.
        """
        # Show model in tensorboard:
        logging.info("Showing model graph in TensorBoard...")
        if is_master():
            plot_model(state.model, folder=self.tensorboard_dir)
        # FIXME(lvdmaaten): Expose folder / SummaryWriter for TensorBoard.
