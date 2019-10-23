#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from test.generic.config_utils import get_fast_test_task_config, get_test_args
from test.generic.utils import compare_model_state

from classy_vision.generic.util import get_checkpoint_dict
from classy_vision.tasks import FineTuningTask, build_task
from classy_vision.trainer import ClassyTrainer


class TestFineTuningTask(unittest.TestCase):
    def _compare_model_state(self, state_1, state_2, check_heads=True):
        return compare_model_state(self, state_1, state_2, check_heads=check_heads)

    def _get_fine_tuning_config(self, head_num_classes=1000):
        config = get_fast_test_task_config(head_num_classes=head_num_classes)
        config["name"] = "fine_tuning"
        config["num_epochs"] = 10
        return config

    def _get_pre_train_config(self, head_num_classes=1000):
        config = get_fast_test_task_config(head_num_classes=head_num_classes)
        config["num_epochs"] = 10
        return config

    def test_build_task(self):
        config = self._get_fine_tuning_config()
        args = get_test_args()
        task = build_task(config, args)
        self.assertIsInstance(task, FineTuningTask)

    def test_prepare(self):
        pre_train_config = self._get_pre_train_config()
        args = get_test_args()
        pre_train_task = build_task(pre_train_config, args)
        pre_train_task.prepare()
        checkpoint = get_checkpoint_dict(pre_train_task, args)

        fine_tuning_config = self._get_fine_tuning_config()
        args = get_test_args()
        fine_tuning_task = build_task(fine_tuning_config, args)
        # cannot prepare a fine tuning task without a pre training checkpoint
        with self.assertRaises(Exception):
            fine_tuning_task.prepare()

        fine_tuning_task.set_pretrained_checkpoint(checkpoint)
        fine_tuning_task.prepare()

        # test a fine tuning task with incompatible heads
        fine_tuning_config = self._get_fine_tuning_config(head_num_classes=10)
        args = get_test_args()
        fine_tuning_task = build_task(fine_tuning_config, args)
        fine_tuning_task.set_pretrained_checkpoint(checkpoint)
        # cannot prepare a fine tuning task with a pre training checkpoint which
        # has incompatible heads
        with self.assertRaises(Exception):
            fine_tuning_task.prepare()

        fine_tuning_task.set_pretrained_checkpoint(checkpoint).set_reset_heads(True)
        fine_tuning_task.prepare()

    def test_train(self):
        pre_train_config = self._get_pre_train_config(head_num_classes=1000)
        args = get_test_args()
        pre_train_task = build_task(pre_train_config, args)
        trainer = ClassyTrainer(use_gpu=False)
        trainer.train(pre_train_task)
        checkpoint = get_checkpoint_dict(pre_train_task, args)

        for reset_heads, heads_num_classes in [(False, 1000), (True, 200)]:
            fine_tuning_config = self._get_fine_tuning_config(
                head_num_classes=heads_num_classes
            )
            fine_tuning_task = build_task(fine_tuning_config, args)
            fine_tuning_task.set_pretrained_checkpoint(
                copy.deepcopy(checkpoint)
            ).set_reset_heads(reset_heads)
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
            accuracy = fine_tuning_task.meters[0].value["top_1"]
            self.assertAlmostEqual(accuracy, 1.0)
