#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from enum import Enum
from typing import Any, Callable, Dict, Union

from classy_vision.generic.util import (
    load_and_broadcast_checkpoint,
    update_classy_model,
)
from classy_vision.tasks import ClassificationTask, register_task


class FreezeUntil(Enum):
    """
    Enum for a pre-specified point to freeze the classy model unitl.

    Attributes:
        HEAD (str): Freeze the model unitl the classy head
    """

    HEAD = "head"

    def __eq__(self, other: str):
        return other.lower() == self.value


@register_task("fine_tuning")
class FineTuningTask(ClassificationTask):
    """Finetuning training task.

    This task encapsultates all of the components and steps needed to
    fine-tune a classifier using a :class:`classy_vision.trainer.ClassyTrainer`.

    :var pretrained_checkpoint_path: String path to pretrained model
    :var reset_heads: bool. Whether or not to reset the model heads during finetuning.
    :var freeze_until: optional string. If specified, must be a string name of a module within
        the model. Finetuning will freeze the model up to this module. Model weights will
        only be trainable from this modeule onwards, always including the head. To freeze the
        trunk model, specify 'head' as the un-freeze point.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_checkpoint_dict = None
        self.pretrained_checkpoint_path = None
        self.pretrained_checkpoint_load_strict = True
        self.reset_heads = False
        self.freeze_until = None

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
            task.set_pretrained_checkpoint(pretrained_checkpoint_path)
            task.set_pretrained_checkpoint_load_strict(
                config.get("pretrained_checkpoint_load_strict", True)
            )

        task.set_reset_heads(config.get("reset_heads", False))
        assert (
            "freeze_trunk" not in config or "freeze_until" not in config
        ), "Config options 'freeze_trunk' and 'freeze_until' cannot both be specified"
        if "freeze_trunk" in config:
            task.set_freeze_trunk(config.get("freeze_trunk", False))
        else:
            task.set_freeze_until(config.get("freeze_until", None))
        return task

    def set_pretrained_checkpoint(self, checkpoint_path: str) -> "FineTuningTask":
        self.pretrained_checkpoint_path = checkpoint_path
        return self

    def set_pretrained_checkpoint_load_strict(
        self, pretrained_checkpoint_load_strict: bool
    ):
        self.pretrained_checkpoint_load_strict = pretrained_checkpoint_load_strict
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
        if freeze_trunk:
            self.freeze_until = FreezeUntil.HEAD.value
            warnings.warn(
                "Congig option freeze_trunk has been deprecated. "
                "Use \"freeze_until:'head'\" instead",
                DeprecationWarning,
            )

        return self

    def set_freeze_until(self, freeze_until: Union[str, None]) -> "FineTuningTask":
        self.freeze_until = freeze_until
        return self

    def _set_model_train_mode(self):
        phase = self.phases[self.phase_idx]
        self.loss.train(phase["train"])

        if self.freeze_until is not None:
            # convert all the sub-modules to the eval mode, except the heads
            self.base_model.eval()
            self._apply_to_nonfrozen(lambda x: x.train(phase["train"]))
        else:
            self.base_model.train(phase["train"])

    def _apply_to_nonfrozen(self, callable: Callable[..., Any]) -> None:
        for heads in self.base_model.get_heads().values():
            for h in heads:
                callable(h)
        if not self.freeze_until == FreezeUntil.HEAD:
            unfrozen_module = False
            for name, module in self.base_model.named_modules():
                if name == self.freeze_until:
                    unfrozen_module = True
                if unfrozen_module:
                    callable(module)
            assert (
                unfrozen_module
            ), f"Freeze until point {self.freeze_until} not found in model"

    def prepare(self) -> None:
        super().prepare()
        if self.checkpoint_dict is None:
            # no checkpoint exists, load the model's state from the pretrained
            # checkpoint

            if self.pretrained_checkpoint_path:
                self.pretrained_checkpoint_dict = load_and_broadcast_checkpoint(
                    self.pretrained_checkpoint_path
                )

            assert (
                self.pretrained_checkpoint_dict is not None
            ), "Need a pretrained checkpoint for fine tuning"

            state_load_success = update_classy_model(
                self.base_model,
                self.pretrained_checkpoint_dict["classy_state_dict"]["base_model"],
                self.reset_heads,
                self.pretrained_checkpoint_load_strict,
            )
            assert (
                state_load_success
            ), "Update classy state from pretrained checkpoint was unsuccessful."

        if self.freeze_until is not None:
            # do not track gradients for all the parameters in the model except
            # for the parameters in the heads
            for param in self.base_model.parameters():
                param.requires_grad = False

            def _set_requires_grad_true(x):
                for param in x.parameters():
                    param.requires_grad = True

            self._apply_to_nonfrozen(_set_requires_grad_true)
            # re-create ddp model
            self.distributed_model = None
            self.init_distributed_data_parallel_model()
