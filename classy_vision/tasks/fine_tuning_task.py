#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

from classy_vision.generic.util import (
    load_and_broadcast_checkpoint,
    update_classy_model,
)
from classy_vision.tasks import ClassificationTask, register_task


@register_task("fine_tuning")
class FineTuningTask(ClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_checkpoint_dict = None
        self.pretrained_checkpoint_path = None
        self.pretrained_checkpoint_load_strict = True
        self.hooks_load_from_pretrained_checkpoint = []
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

        pretrained_checkpoint_path = config.get("pretrained_checkpoint")
        if pretrained_checkpoint_path:
            task.set_pretrained_checkpoint(
                pretrained_checkpoint_path
            ).set_pretrained_checkpoint_load_strict(
                config.get("pretrained_checkpoint_load_strict", True)
            ).set_hooks_load_from_pretrained_checkpoint(
                config.get("hooks_load_from_pretrained_checkpoint", [])
            )

        task.set_reset_heads(config.get("reset_heads", False))
        task.set_freeze_trunk(config.get("freeze_trunk", False))
        return task

    def set_pretrained_checkpoint(self, checkpoint_path: str) -> "FineTuningTask":
        self.pretrained_checkpoint_path = checkpoint_path
        return self

    def set_pretrained_checkpoint_load_strict(
        self, pretrained_checkpoint_load_strict: bool
    ):
        self.pretrained_checkpoint_load_strict = pretrained_checkpoint_load_strict
        return self

    def set_hooks_load_from_pretrained_checkpoint(
        self, hooks_load_from_pretrained_checkpoint: List[str]
    ):
        """
        Args:
            hooks_load_from_pretrained_checkpoint: a list of the names of the hooks that we
                want to load state dict from pretrained checkpoint
        """
        self.hooks_load_from_pretrained_checkpoint = (
            hooks_load_from_pretrained_checkpoint
        )
        return self

    def _set_pretrained_checkpoint_dict(
        self, checkpoint_dict: Dict[str, Any]
    ) -> "FineTuningTask":
        self.pretrained_checkpoint_dict = checkpoint_dict
        return self

    def set_reset_heads(self, reset_heads: bool) -> "FineTuningTask":
        self.reset_heads = reset_heads
        return self

    def set_freeze_trunk(self, freeze_trunk: bool) -> "FineTuningTask":
        self.freeze_trunk = freeze_trunk
        return self

    def _set_model_train_mode(self):
        phase = self.phases[self.phase_idx]
        self.loss.train(phase["train"])

        if self.freeze_trunk:
            # convert all the sub-modules to the eval mode, except the heads
            self.base_model.eval()
            for heads in self.base_model.get_heads().values():
                for h in heads:
                    h.train(phase["train"])
        else:
            self.base_model.train(phase["train"])

    def _load_hooks_from_pretrained_checkpoint(self, state: Dict[str, Any]):
        for hook in self.hooks:
            if (
                hook.name() in state["hooks"]
                and hook.name() in self.hooks_load_from_pretrained_checkpoint
            ):
                hook.set_classy_state(state["hooks"][hook.name()])

    def prepare(self) -> None:
        super().prepare()
        if self.checkpoint_dict is None:
            # no checkpoint exists, load the model's state from the pretrained
            # checkpoint

            if self.pretrained_checkpoint_path:
                self.pretrained_checkpoint_dict = load_and_broadcast_checkpoint(
                    self.pretrained_checkpoint_path
                )

            if self.pretrained_checkpoint_dict is None:
                logging.warn("a pretrained checkpoint is not provided")
            else:
                assert (
                    self.pretrained_checkpoint_dict is not None
                ), "Need a pretrained checkpoint for fine tuning"

                state = self.pretrained_checkpoint_dict["classy_state_dict"]

                state_load_success = update_classy_model(
                    self.base_model,
                    state["base_model"],
                    self.reset_heads,
                    self.pretrained_checkpoint_load_strict,
                )
                assert (
                    state_load_success
                ), "Update classy state from pretrained checkpoint was unsuccessful."

                self._load_hooks_from_pretrained_checkpoint(state)

        if self.freeze_trunk:
            # do not track gradients for all the parameters in the model except
            # for the parameters in the heads
            for param in self.base_model.parameters():
                param.requires_grad = False
            for heads in self.base_model.get_heads().values():
                for h in heads:
                    for param in h.parameters():
                        param.requires_grad = True
            # re-create ddp model
            self.distributed_model = None
            self.init_distributed_data_parallel_model()
