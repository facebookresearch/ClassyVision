#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from classy_vision.generic.util import load_checkpoint, update_classy_model
from classy_vision.tasks import ClassificationTask, register_task


@register_task("fine_tuning")
class FineTuningTask(ClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_checkpoint = None
        self.reset_heads = False
        self.freeze_trunk = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FineTuningTask":
        """Instantiates a FineTuningTask from a configuration.

        Args:
            config: A configuration for a FineTuningTask.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A FineTuningTask instance.
        """
        task = super().from_config(config)

        pretrained_checkpoint = load_checkpoint(config.get("pretrained_checkpoint"))

        if pretrained_checkpoint is not None:
            task.set_pretrained_checkpoint(pretrained_checkpoint)

        task.set_reset_heads(config.get("reset_heads", False))
        task.set_freeze_trunk(config.get("freeze_trunk", False))
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

    def set_freeze_trunk(self, freeze_trunk: bool) -> "FineTuningTask":
        self.freeze_trunk = freeze_trunk
        return self

    def _set_model_train_mode(self):
        phase = self.phases[self.phase_idx]
        if self.freeze_trunk:
            # convert all the sub-modules to the eval mode, except the heads
            self.base_model.eval()
            for heads in self.base_model.get_heads().values():
                for h in heads.values():
                    h.train(phase["train"])
        else:
            self.base_model.train(phase["train"])

    def prepare(
        self,
        num_dataloader_workers: int = 0,
        pin_memory: bool = False,
        use_gpu: bool = False,
        dataloader_mp_context=None,
    ) -> None:
        assert (
            self.pretrained_checkpoint is not None
        ), "Need a pretrained checkpoint for fine tuning"
        super().prepare(
            num_dataloader_workers, pin_memory, use_gpu, dataloader_mp_context
        )
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

        if self.freeze_trunk:
            # do not track gradients for all the parameters in the model except
            # for the parameters in the heads
            for param in self.base_model.parameters():
                param.requires_grad = False
            for heads in self.base_model.get_heads().values():
                for h in heads.values():
                    for param in h.parameters():
                        param.requires_grad = True
            # re-create ddp model
            self.distributed_model = None
            self.init_distributed_data_parallel_model()
