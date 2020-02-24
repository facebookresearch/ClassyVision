#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Collection, Dict, Optional

from classy_vision import tasks
from classy_vision.generic.distributed_util import is_master
from classy_vision.generic.util import get_checkpoint_dict, save_checkpoint
from classy_vision.hooks.classy_hook import ClassyHook
from fvcore.common.file_io import PathManager


class CheckpointHook(ClassyHook):
    """
    Hook to checkpoint a model's task.

    Saves the checkpoints in checkpoint_folder.
    """

    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_step = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(
        self,
        checkpoint_folder: str,
        input_args: Any,
        phase_types: Optional[Collection[str]] = None,
        checkpoint_period: int = 1,
    ) -> None:
        """The constructor method of CheckpointHook.

        Args:
            checkpoint_folder: Folder to store checkpoints in
            input_args: Any arguments to save about the runtime setup. For example,
                it is useful to store the config that was used to instantiate the model.
            phase_types: If `phase_types` is specified, only checkpoint on those phase
                types. Each item in `phase_types` must be either "train" or "test". If
                not specified, it is set to checkpoint after "train" phases.
            checkpoint_period: Checkpoint at the end of every x phases (default 1)
        """
        super().__init__()
        self.checkpoint_folder: str = checkpoint_folder
        self.input_args: Any = input_args
        if phase_types is None:
            phase_types = ["train"]
        assert len(phase_types) > 0 and all(
            phase_type in ["train", "test"] for phase_type in phase_types
        ), "phase_types should contain one or more of ['train', 'test']"
        assert (
            isinstance(checkpoint_period, int) and checkpoint_period > 0
        ), "checkpoint period must be positive"

        self.phase_types: Collection[str] = phase_types
        self.checkpoint_period: int = checkpoint_period
        self.phase_counter: int = 0

    def _save_checkpoint(self, task, filename):
        if getattr(task, "test_only", False):
            return
        assert PathManager.exists(
            self.checkpoint_folder
        ), "Checkpoint folder '{}' deleted unexpectedly".format(self.checkpoint_folder)

        # save checkpoint:
        logging.info("Saving checkpoint to '{}'...".format(self.checkpoint_folder))
        checkpoint_file = save_checkpoint(
            self.checkpoint_folder, get_checkpoint_dict(task, self.input_args)
        )

        # make copy of checkpoint that won't be overwritten:
        if checkpoint_file:
            PathManager.copy(checkpoint_file, f"{self.checkpoint_folder}/{filename}")

    def on_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        if not is_master() or getattr(task, "test_only", False):
            return
        if not PathManager.exists(self.checkpoint_folder):
            err_msg = "Checkpoint folder '{}' does not exist.".format(
                self.checkpoint_folder
            )
            raise FileNotFoundError(err_msg)

    def on_phase_end(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Checkpoint the task every checkpoint_period phases.

        We do not necessarily checkpoint the task at the end of every phase.
        """
        if not is_master() or task.phase_type not in self.phase_types:
            return

        self.phase_counter += 1
        if self.phase_counter % self.checkpoint_period != 0:
            return

        checkpoint_name = "model_phase-{phase}_end.torch".format(phase=task.phase_idx)
        self._save_checkpoint(task, checkpoint_name)
