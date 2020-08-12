#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from test.generic.config_utils import get_test_mlp_task_config
from test.generic.hook_test_utils import HookTestBase

import torch
import torch.nn as nn
from classy_vision.hooks import ClassyHook
from classy_vision.hooks.precise_batch_norm_hook import PreciseBatchNormHook
from classy_vision.tasks import build_task
from classy_vision.trainer import ClassyTrainer


class TestPreciseBatchNormHook(HookTestBase):
    def _get_bn_stats(self, model):
        model = copy.deepcopy(model)
        stats = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                stats[name] = {"mean": module.running_mean, "var": module.running_var}
        return stats

    def _compare_bn_stats(self, stats_1, stats_2):
        # make sure the stats are non empty
        self.assertGreater(len(stats_1), 0)
        for name in stats_1:
            if not torch.allclose(
                stats_1[name]["mean"], stats_2[name]["mean"]
            ) or not torch.allclose(stats_1[name]["var"], stats_2[name]["var"]):
                return False
        return True

    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        self.constructor_test_helper(
            config={"num_samples": 10},
            hook_type=PreciseBatchNormHook,
            hook_registry_name="precise_bn",
            invalid_configs=[{}, {"num_samples": 0}],
        )

    def test_train(self):
        config = get_test_mlp_task_config()
        task = build_task(config)
        num_samples = 10
        precise_batch_norm_hook = PreciseBatchNormHook(num_samples)
        task.set_hooks([precise_batch_norm_hook])
        task.prepare()
        trainer = ClassyTrainer()
        trainer.train(task)

    def test_bn_stats(self):
        base_self = self

        class TestHook(ClassyHook):
            on_start = ClassyHook._noop
            on_phase_start = ClassyHook._noop
            on_phase_end = ClassyHook._noop
            on_end = ClassyHook._noop

            def __init__(self):
                self.train_bn_stats = None
                self.test_bn_stats = None

            def on_step(self, task):
                if task.train:
                    self.train_bn_stats = base_self._get_bn_stats(task.base_model)
                else:
                    self.test_bn_stats = base_self._get_bn_stats(task.base_model)

        config = get_test_mlp_task_config()
        task = build_task(config)
        num_samples = 10
        precise_batch_norm_hook = PreciseBatchNormHook(num_samples)
        test_hook = TestHook()
        task.set_hooks([precise_batch_norm_hook, test_hook])
        trainer = ClassyTrainer()
        trainer.train(task)

        updated_bn_stats = self._get_bn_stats(task.base_model)

        # the stats should be modified after train steps but not after test steps
        self.assertFalse(
            self._compare_bn_stats(test_hook.train_bn_stats, updated_bn_stats)
        )
        self.assertTrue(
            self._compare_bn_stats(test_hook.test_bn_stats, updated_bn_stats)
        )
