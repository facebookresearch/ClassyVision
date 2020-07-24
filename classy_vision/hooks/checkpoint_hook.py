#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Collection, Dict, Optional  # noqa

from classy_vision.generic.distributed_util import is_primary
from classy_vision.generic.util import get_checkpoint_dict, save_checkpoint
from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook
from fvcore.common.file_io import PathManager


gfs_prefix_list = {
    "/mnt/gfsdataswarm",
    "/mnt/gfsdataswarm-global",
    "/mnt/vol",
    "/mnt/shared",
    "/mnt/homedir",
}


@register_hook("checkpoint")
class CheckpointHook(ClassyHook):
    """
    Hook to checkpoint a model's task.

    Saves the checkpoints in checkpoint_folder.
    """

    on_phase_start = ClassyHook._noop
    on_step = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(
        self,
        checkpoint_folder: str,
        input_args: Any = None,
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
        assert isinstance(
            checkpoint_folder, str
        ), "checkpoint_folder must be a string specifying the checkpoint directory"
        assert (
            isinstance(checkpoint_period, int) and checkpoint_period > 0
        ), "checkpoint_period must be a positive integer"

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

    @classmethod
    def get_checkpoint_name(cls, phase_idx):
        return "model_phase-{phase}_end.torch".format(phase=phase_idx)

    def _save_checkpoint(self, task, filename):
        if getattr(task, "test_only", False):
            return
        assert PathManager.exists(
            self.checkpoint_folder
        ), "Checkpoint folder '{}' deleted unexpectedly".format(self.checkpoint_folder)

        for prefix in gfs_prefix_list:
            if self.checkpoint_folder.startswith(prefix):
                logging.warning(
                    "GFS is deprecating... please save checkpoint to manifold!"
                )
                break

        # save checkpoint:
        logging.info("Saving checkpoint to '{}'...".format(self.checkpoint_folder))
        checkpoint_file = save_checkpoint(
            self.checkpoint_folder, get_checkpoint_dict(task, self.input_args)
        )

        # make copy of checkpoint that won't be overwritten:
        PathManager.copy(checkpoint_file, f"{self.checkpoint_folder}/{filename}")

    def on_start(self, task) -> None:
        if not is_primary() or getattr(task, "test_only", False):
            return
        if not PathManager.exists(self.checkpoint_folder):
            err_msg = "Checkpoint folder '{}' does not exist.".format(
                self.checkpoint_folder
            )
            raise FileNotFoundError(err_msg)

    def on_phase_end(self, task) -> None:
        """Checkpoint the task every checkpoint_period phases.

        We do not necessarily checkpoint the task at the end of every phase.
        """
        if not is_primary() or task.phase_type not in self.phase_types:
            return

        self.phase_counter += 1
        if self.phase_counter % self.checkpoint_period != 0:
            return

        checkpoint_name = CheckpointHook.get_checkpoint_name(task.phase_idx)
        self._save_checkpoint(task, checkpoint_name)
