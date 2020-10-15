#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from test.generic.config_utils import get_fast_test_task_config
from test.generic.utils import compare_model_state
from typing import Any, Dict
from unittest import mock

import torch
import torch.nn as nn
from classy_vision.generic.util import get_checkpoint_dict
from classy_vision.losses import ClassyLoss, register_loss
from classy_vision.tasks import FineTuningTask, build_task
from classy_vision.trainer import LocalTrainer
from torch.nn.modules.loss import CrossEntropyLoss


@register_loss("batchnorm_cross_entropy_loss")
class BatchNormCrossEntropyLoss(ClassyLoss):
    """A special loss containing a BatchNorm module"""

    def __init__(self, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_classes)
        self.fc = nn.Linear(num_classes, num_classes)
        self.xent = CrossEntropyLoss()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BatchNormCrossEntropyLoss":
        assert "num_classes" in config
        return cls(config["num_classes"])

    def forward(self, x, target):
        return self.xent(self.fc(self.bn(x)), target)


class TestFineTuningTask(unittest.TestCase):
    def _compare_model_state(self, state_1, state_2, check_heads=True):
        return compare_model_state(self, state_1, state_2, check_heads=check_heads)

    def _compare_state_dict(self, state_1, state_2, check_heads=True):
        for k in state_1.keys():
            self.assertTrue(torch.allclose(state_1[k].cpu(), state_2[k].cpu()))

    def _get_fine_tuning_config(
        self, head_num_classes=100, pretrained_checkpoint=False
    ):
        config = get_fast_test_task_config(head_num_classes=head_num_classes)
        config["name"] = "fine_tuning"
        config["num_epochs"] = 2

        if pretrained_checkpoint:
            config["pretrained_checkpoint"] = "/path/to/pretrained/checkpoint"

        return config

    def _get_pre_train_config(self, head_num_classes=100):
        config = get_fast_test_task_config(head_num_classes=head_num_classes)
        config["num_epochs"] = 2
        return config

    def test_build_task(self):
        config = self._get_fine_tuning_config()
        task = build_task(config)
        self.assertIsInstance(task, FineTuningTask)

        config = self._get_fine_tuning_config(pretrained_checkpoint=True)

        with mock.patch("classy_vision.tasks.FineTuningTask.set_pretrained_checkpoint"):
            task = build_task(config)

        self.assertIsInstance(task, FineTuningTask)

    def test_prepare(self):
        pre_train_config = self._get_pre_train_config()
        pre_train_task = build_task(pre_train_config)
        pre_train_task.prepare()
        checkpoint = get_checkpoint_dict(pre_train_task, {})

        fine_tuning_config = self._get_fine_tuning_config()
        fine_tuning_task = build_task(fine_tuning_config)

        # test: cannot prepare a fine tuning task without a pre-trained checkpoint
        with self.assertRaises(Exception):
            fine_tuning_task.prepare()

        # test: prepare should succeed after pre-trained checkpoint is set
        fine_tuning_task = build_task(fine_tuning_config)
        fine_tuning_task._set_pretrained_checkpoint_dict(checkpoint)
        fine_tuning_task.prepare()

        # test: prepare should succeed if a pre-trained checkpoint is provided in the
        # config
        fine_tuning_config = self._get_fine_tuning_config(pretrained_checkpoint=True)
        fine_tuning_task = build_task(fine_tuning_config)
        with mock.patch(
            "classy_vision.tasks.fine_tuning_task.load_and_broadcast_checkpoint",
            return_value=checkpoint,
        ):
            fine_tuning_task.prepare()

        # test: a fine tuning task with incompatible heads with a manually set
        # pre-trained checkpoint should fail to prepare if the heads are not reset
        fine_tuning_config = self._get_fine_tuning_config(head_num_classes=10)
        fine_tuning_task = build_task(fine_tuning_config)
        fine_tuning_task._set_pretrained_checkpoint_dict(checkpoint)

        with self.assertRaises(Exception):
            fine_tuning_task.prepare()

        # test: a fine tuning task with incompatible heads with a manually set
        # pre-trained checkpoint should succeed to prepare if the heads are reset
        fine_tuning_task = build_task(fine_tuning_config)
        fine_tuning_task._set_pretrained_checkpoint_dict(
            copy.deepcopy(checkpoint)
        ).set_reset_heads(True)

        fine_tuning_task.prepare()

        # test: a fine tuning task with incompatible heads with the pre-trained
        # checkpoint provided in the config should fail to prepare
        fine_tuning_config = self._get_fine_tuning_config(
            head_num_classes=10, pretrained_checkpoint=True
        )

        fine_tuning_task = build_task(fine_tuning_config)
        with mock.patch(
            "classy_vision.tasks.fine_tuning_task.load_and_broadcast_checkpoint",
            return_value=copy.deepcopy(checkpoint),
        ) and self.assertRaises(Exception):
            fine_tuning_task.prepare()

        # test: a fine tuning task with incompatible heads with the pre-trained
        # checkpoint provided in the config should succeed to prepare if the heads are
        # reset
        fine_tuning_task = build_task(fine_tuning_config)
        fine_tuning_task.set_reset_heads(True)
        with mock.patch(
            "classy_vision.tasks.fine_tuning_task.load_and_broadcast_checkpoint",
            return_value=copy.deepcopy(checkpoint),
        ):
            fine_tuning_task.prepare()

    def test_train(self):
        pre_train_config = self._get_pre_train_config(head_num_classes=100)
        pre_train_task = build_task(pre_train_config)
        trainer = LocalTrainer()
        trainer.train(pre_train_task)
        checkpoint = get_checkpoint_dict(pre_train_task, {})

        for reset_heads, heads_num_classes in [(False, 100), (True, 20)]:
            for freeze_trunk in [True, False]:
                fine_tuning_config = self._get_fine_tuning_config(
                    head_num_classes=heads_num_classes
                )
                fine_tuning_task = build_task(fine_tuning_config)
                fine_tuning_task = (
                    fine_tuning_task._set_pretrained_checkpoint_dict(
                        copy.deepcopy(checkpoint)
                    )
                    .set_reset_heads(reset_heads)
                    .set_freeze_trunk(freeze_trunk)
                )
                # run in test mode to compare the model state
                fine_tuning_task.set_test_only(True)
                trainer.train(fine_tuning_task)
                self._compare_model_state(
                    pre_train_task.model.get_classy_state(),
                    fine_tuning_task.model.get_classy_state(),
                    check_heads=not reset_heads,
                )
                # run in train mode to check accuracy
                fine_tuning_task.set_test_only(False)
                trainer.train(fine_tuning_task)
                if freeze_trunk:
                    # if trunk is frozen the states should be the same
                    self._compare_model_state(
                        pre_train_task.model.get_classy_state(),
                        fine_tuning_task.model.get_classy_state(),
                        check_heads=False,
                    )
                else:
                    # trunk isn't frozen, the states should be different
                    with self.assertRaises(Exception):
                        self._compare_model_state(
                            pre_train_task.model.get_classy_state(),
                            fine_tuning_task.model.get_classy_state(),
                            check_heads=False,
                        )

                accuracy = fine_tuning_task.meters[0].value["top_1"]
                self.assertAlmostEqual(accuracy, 1.0)

    def test_train_parametric_loss(self):
        heads_num_classes = 100
        pre_train_config = self._get_pre_train_config(
            head_num_classes=heads_num_classes
        )
        pre_train_config["loss"] = {
            "name": "batchnorm_cross_entropy_loss",
            "num_classes": heads_num_classes,
        }
        pre_train_task = build_task(pre_train_config)
        trainer = LocalTrainer()
        trainer.train(pre_train_task)
        checkpoint = get_checkpoint_dict(pre_train_task, {})

        fine_tuning_config = self._get_fine_tuning_config(
            head_num_classes=heads_num_classes
        )
        fine_tuning_config["loss"] = {
            "name": "batchnorm_cross_entropy_loss",
            "num_classes": heads_num_classes,
        }

        fine_tuning_task = build_task(fine_tuning_config)
        fine_tuning_task._set_pretrained_checkpoint_dict(copy.deepcopy(checkpoint))
        # run in test mode to compare the loss state. Since we have a BatchNorm module in
        # the loss, its moving mean/std should be unchanged when we run in test-only mode
        fine_tuning_task.set_test_only(True)
        loss_state = copy.deepcopy(fine_tuning_task.loss.get_classy_state())
        trainer.train(fine_tuning_task)
        self._compare_state_dict(loss_state, fine_tuning_task.loss.get_classy_state())
