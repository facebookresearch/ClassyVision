#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook
from fvcore.common.file_io import PathManager


DEFAULT_FILE_NAME = "predictions.csv"


@register_hook("output_csv")
class OutputCSVHook(ClassyHook):
    on_phase_start = ClassyHook._noop
    on_start = ClassyHook._noop

    def __init__(self, folder, id_key="id", delimiter="\t") -> None:
        super().__init__()
        self.output_path = f"{folder}/{DEFAULT_FILE_NAME}"
        self.file = PathManager.open(self.output_path, "a")
        self.id_key = id_key
        self.delimiter = delimiter

    def on_start(self, task) -> None:
        # File header
        self.file.write(
            self.delimiter.join(["sample_id", "prediction", "target"]) + "\n"
        )

    def on_step(self, task) -> None:
        """Saves the output of the model to a CSV file.

        This hook assumes the dataset provides an "id" key. It also expects the
        task to provide an output of shape (B, C) where B is the batch size and
        C is the number of classes. Targets can be either one-hot encoded or
        single numbers."""

        if self.id_key not in task.last_batch.sample:
            return

        if task.train:
            return

        assert (
            len(task.last_batch.output.shape) == 2
        ), "First dimension must be batch size, second is the class logits"
        assert len(task.last_batch.sample["target"].shape) in [
            1,
            2,
        ], "Target must be integer or one-hot encoded vectors"

        sample_ids = task.last_batch.sample[self.id_key].tolist()
        predictions = task.last_batch.output.argmax(dim=1).tolist()
        target = task.last_batch.sample["target"]
        # One-hot encoded vectors
        if len(target.shape) == 2:
            targets = target.argmax(dim=1)
        targets = target.tolist()

        for sample_id, prediction, target in zip(sample_ids, predictions, targets):
            self.file.write(
                self.delimiter.join([str(sample_id), str(prediction), str(target)])
                + "\n"
            )

    def on_phase_end(self, task) -> None:
        self.file.flush()

    def on_end(self, task) -> None:
        self.file.close()
