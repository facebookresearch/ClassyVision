#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from classy_vision import tasks
from classy_vision.generic.distributed_util import is_master
from classy_vision.hooks.classy_hook import ClassyHook


try:
    import progressbar

    progressbar_available = True
except ImportError:
    progressbar_available = False


class ProgressBarHook(ClassyHook):
    """
    Displays a progress bar to show progress in processing batches.
    """

    on_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self) -> None:
        """The constructor method of ProgressBarHook."""
        super().__init__()
        self.progress_bar: Optional[progressbar.ProgressBar] = None
        self.bar_size: int = 0
        self.batches: int = 0

    def on_phase_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Create and display a progress bar with 0 progress."""
        if not progressbar_available:
            raise RuntimeError(
                "progressbar module not installed, cannot use ProgressBarHook"
            )

        if is_master():
            self.bar_size = task.num_batches_per_phase
            self.batches = 0
            self.progress_bar = progressbar.ProgressBar(self.bar_size)
            self.progress_bar.start()

    def on_step(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Update the progress bar with the batch size."""
        if task.train and is_master() and self.progress_bar is not None:
            self.batches += 1
            self.progress_bar.update(min(self.batches, self.bar_size))

    def on_phase_end(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Clear the progress bar at the end of the phase."""
        if is_master() and self.progress_bar is not None:
            self.progress_bar.finish()
