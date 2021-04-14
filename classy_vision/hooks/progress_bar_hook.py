#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from classy_vision.generic.distributed_util import is_primary
from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook


try:
    import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False


@register_hook("progress_bar")
class ProgressBarHook(ClassyHook):
    """
    Displays progress bars to show progress in processing batches.

    The permanent main progress bar tracks the overall progress in the main task.
    The nested progress bar tracks the progress in the current phase.

    This hook assumes that the task passed as argument contains the
    following fields (e.g. ``classy_vision.tasks.ClassificationTask``):

    - ``phases``: a list of train and test phases
    - ``last_batch``: to access the last labels
    """

    def __init__(self) -> None:
        """The constructor method of ProgressBarHook."""
        super().__init__()
        self.progress_bar: Optional[tqdm.tqdm] = None
        self.phase_bar: Optional[tqdm.tqdm] = None

    def on_start(self, task) -> None:
        """Create and display a progress bar with 0 progress."""
        if not tqdm_available:
            raise RuntimeError("tqdm module not installed, cannot use ProgressBarHook")
        if is_primary():
            # Compute the total number of images processed
            total_images = 0
            for phase in task.phases:
                phase_type = "train" if phase["train"] else "test"
                total_images += len(task.datasets[phase_type])
            # Create the main task progress bar
            self.progress_bar = tqdm.tqdm(
                total=total_images, desc="task", unit="images"
            )

    def on_phase_start(self, task) -> None:
        if is_primary():
            phase_images = len(task.datasets[task.phase_type])
            self.phase_bar = tqdm.tqdm(
                total=phase_images, desc=task.phase_type, unit="images", leave=False
            )

    def on_step(self, task) -> None:
        """Update the progress bar with the batch size."""
        if is_primary():
            batch_size = task.last_batch.output.size(0)
            if self.progress_bar is not None:
                self.progress_bar.update(batch_size)
            if self.phase_bar is not None:
                self.phase_bar.update(batch_size)

    def on_phase_end(self, task) -> None:
        """Clear the progress bar at the end of the phase."""
        if is_primary() and self.phase_bar is not None:
            self.phase_bar.close()

    def on_end(self, task) -> None:
        """Clear the progress bar at the end of the task."""
        if is_primary() and self.progress_bar is not None:
            self.progress_bar.close()
