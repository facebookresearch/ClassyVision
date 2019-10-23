#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from classy_vision.generic.util import update_classy_model
from classy_vision.tasks import ClassificationTask, register_task


@register_task("fine_tuning")
class FineTuningTask(ClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_checkpoint = None
        self.reset_heads = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FineTuningTask":
        task = super().from_config(config)
        task.reset_heads = config.get("reset_heads", False)
        return task

    def set_pretrained_checkpoint(self, checkpoint: Dict[str, Any]) -> "FineTuningTask":
        assert (
            "classy_state_dict" in checkpoint
        ), "Checkpoint does not contain classy_state_dict"
        self.pretrained_checkpoint = checkpoint
        return self

    def set_reset_heads(self, reset_heads: bool) -> "FineTuningTask":
        self.reset_heads = reset_heads
        return self

    def prepare(
        self, num_workers: int = 0, pin_memory: bool = False, use_gpu: bool = False
    ) -> None:
        assert (
            self.pretrained_checkpoint is not None
        ), "Need a pretrained checkpoint for fine tuning"
        super().prepare(num_workers, pin_memory, use_gpu)
        if self.checkpoint is None:
            # no checkpoint exists, load the model's state from the pretrained
            # checkpoint
            state_load_success = update_classy_model(
                self.base_model,
                self.pretrained_checkpoint["classy_state_dict"]["base_model"],
                self.reset_heads,
            )
            assert (
                state_load_success
            ), "Update classy state from pretrained checkpoint was unsuccessful."
