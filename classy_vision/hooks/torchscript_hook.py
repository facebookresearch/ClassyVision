#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from classy_vision.generic.distributed_util import is_primary
from classy_vision.generic.util import eval_model, get_model_dummy_input
from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook
from fvcore.common.file_io import PathManager


# constants
TORCHSCRIPT_FILE = "torchscript.pt"


@register_hook("torchscript")
class TorchscriptHook(ClassyHook):
    """
    Hook to convert a task model into torch script.

    Saves the torch scripts in torchscript_folder.
    """

    on_phase_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_step = ClassyHook._noop

    def __init__(
        self, torchscript_folder: str, use_trace: bool = True, device: str = "cpu"
    ) -> None:
        """The constructor method of TorchscriptHook.

        Args:
            torchscript_folder: Folder to store torch scripts in.
            use_trace: set to true for tracing and false for scripting,
            device: move to device before saving.
        """
        super().__init__()
        assert isinstance(
            torchscript_folder, str
        ), "torchscript_folder must be a string specifying the torchscript directory"

        self.torchscript_folder: str = torchscript_folder
        self.use_trace: bool = use_trace
        self.device: str = device

    def torchscript_using_trace(self, model):
        input_shape = model.input_shape if hasattr(model, "input_shape") else None
        if not input_shape:
            logging.warning(
                "This model doesn't implement input_shape."
                "Cannot save torchscripted model."
            )
            return
        input_data = get_model_dummy_input(
            model,
            input_shape,
            input_key=model.input_key if hasattr(model, "input_key") else None,
        )
        with eval_model(model) and torch.no_grad():
            torchscript = torch.jit.trace(model, input_data)
        return torchscript

    def torchscript_using_script(self, model):
        with eval_model(model) and torch.no_grad():
            torchscript = torch.jit.script(model)
        return torchscript

    def save_torchscript(self, task) -> None:
        model = task.base_model
        torchscript = (
            self.torchscript_using_trace(model)
            if self.use_trace
            else self.torchscript_using_script(model)
        )

        # save torchscript:
        logging.info("Saving torchscript to '{}'...".format(self.torchscript_folder))
        torchscript = torchscript.to(self.device)
        torchscript_name = f"{self.torchscript_folder}/{TORCHSCRIPT_FILE}"
        with PathManager.open(torchscript_name, "wb") as f:
            torch.jit.save(torchscript, f)

    def on_start(self, task) -> None:
        if not is_primary() or getattr(task, "test_only", False):
            return
        if not PathManager.exists(self.torchscript_folder):
            err_msg = "Torchscript folder '{}' does not exist.".format(
                self.torchscript_folder
            )
            raise FileNotFoundError(err_msg)

    def on_end(self, task) -> None:
        """Save model into torchscript by the end of training."""
        if not is_primary() or getattr(task, "test_only", False):
            return
        self.save_torchscript(task)
