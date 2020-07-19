#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from classy_vision.generic.util import (
    load_and_broadcast_checkpoint,
    split_batchnorm_params,
    update_classy_model,
)
from classy_vision.tasks import ClassificationTask, register_task


def _exclude_parameters(params, params_to_exclude):
    params_to_exclude_ids = [id(p) for p in params_to_exclude]
    params = [p for p in params if id(p) not in params_to_exclude_ids]
    return params


@register_task("fine_tuning")
class FineTuningTask(ClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_checkpoint_dict = None
        self.pretrained_checkpoint_path = None
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
            task.set_pretrained_checkpoint(pretrained_checkpoint_path)

        task.set_reset_heads(config.get("reset_heads", False))
        task.set_freeze_trunk(config.get("freeze_trunk", False))
        task.set_head_lr_scale(config.get("head_lr_scale", 1.0))
        return task

    def set_pretrained_checkpoint(self, checkpoint_path: str) -> "FineTuningTask":
        self.pretrained_checkpoint_path = checkpoint_path
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

    def set_head_lr_scale(self, head_lr_scale: float) -> "FineTuningTask":
        self.head_lr_scale = head_lr_scale
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

    def prepare_optimizer(self, optimizer, model, loss=None):
        if not self.bn_weight_decay:
            model_bn_params, model_params = split_batchnorm_params(model)
            heads = model.get_heads()
            heads_bn_parmas, heads_params = [], []
            for _, block_heads in heads.items():
                for head in block_heads:
                    head_bn_params, head_params = split_batchnorm_params(head)
                    heads_bn_parmas += head_bn_params
                    heads_params += head_params

            model_bn_params = _exclude_parameters(model_bn_params, heads_bn_parmas)
            model_params = _exclude_parameters(model_params, heads_params)
            if loss is not None:
                bn_params_loss, params_loss = split_batchnorm_params(loss)
                heads_bn_parmas += bn_params_loss
                heads_params += params_loss
            frozen_param_groups = []
            if len(model_bn_params) > 0:
                frozen_param_groups.append(
                    {"params": model_bn_params, "weight_decay": 0}
                )
            if len(heads_bn_parmas) > 0:
                frozen_param_groups.append(
                    {
                        "params": heads_bn_parmas,
                        "weight_decay": 0,
                        "lr_scale": self.head_lr_scale,
                    }
                )
            param_groups = [
                {"params": model_params},
                {"params": heads_params, "lr_scale": self.head_lr_scale},
            ]
        else:
            frozen_param_groups = None
            model_params = model.parameters()
            heads_params = []
            heads = model.get_heads()
            for _, block_heads in heads.items():
                for head in block_heads:
                    heads_params += head.parameters()
            model_params = _exclude_parameters(model_params, heads_params)
            if loss is not None:
                heads_params += loss.parameters()

            param_groups = [
                {"params": model_params},
                {"params": heads_params, "lr_scale": self.head_lr_scale},
            ]

        self.optimizer.set_param_groups(
            param_groups=param_groups, frozen_param_groups=frozen_param_groups
        )

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
                for h in heads:
                    for param in h.parameters():
                        param.requires_grad = True
            # re-create ddp model
            self.distributed_model = None
            self.init_distributed_data_parallel_model()
