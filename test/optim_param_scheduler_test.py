#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import Mock

from classy_vision.dataset import build_dataset
from classy_vision.hooks import ClassyHook
from classy_vision.losses import build_loss
from classy_vision.models import build_model
from classy_vision.optim import build_optimizer
from classy_vision.optim.param_scheduler import (
    ClassyParamScheduler,
    UpdateInterval,
    register_param_scheduler,
)
from classy_vision.tasks import ClassificationTask, ClassyTask
from classy_vision.trainer import LocalTrainer


@register_param_scheduler("test_scheduler_where")
class TestParamSchedulerWhere(ClassyParamScheduler):
    def __init__(self):
        self.update_interval = UpdateInterval.STEP

    def __call__(self, where):
        return where

    @classmethod
    def from_config(cls, cfg):
        return cls()


@register_param_scheduler("test_scheduler_where_double")
class TestParamSchedulerWhereDouble(ClassyParamScheduler):
    def __init__(self):
        self.update_interval = UpdateInterval.EPOCH

    def __call__(self, where):
        return where * 2

    @classmethod
    def from_config(cls, cfg):
        return cls()


class TestParamSchedulerIntegration(unittest.TestCase):
    def _get_optimizer_config(self, skip_param_schedulers=False):
        optimizer_config = {"name": "sgd", "num_epochs": 10, "momentum": 0.9}
        if not skip_param_schedulers:
            optimizer_config["param_schedulers"] = {
                "lr": {"name": "test_scheduler_where"},
                "weight_decay": {"name": "test_scheduler_where_double"},
            }
        return optimizer_config

    def _get_config(self, skip_param_schedulers=False):
        return {
            "loss": {"name": "CrossEntropyLoss"},
            "dataset": {
                "train": {
                    "name": "synthetic_image",
                    "split": "train",
                    "num_classes": 2,
                    "crop_size": 20,
                    "class_ratio": 0.5,
                    "num_samples": 10,
                    "seed": 0,
                    "batchsize_per_replica": 5,
                    "use_shuffle": True,
                    "transforms": [
                        {
                            "name": "apply_transform_to_key",
                            "transforms": [
                                {"name": "ToTensor"},
                                {
                                    "name": "Normalize",
                                    "mean": [0.485, 0.456, 0.406],
                                    "std": [0.229, 0.224, 0.225],
                                },
                            ],
                            "key": "input",
                        }
                    ],
                },
                "test": {
                    "name": "synthetic_image",
                    "split": "test",
                    "num_classes": 2,
                    "crop_size": 20,
                    "class_ratio": 0.5,
                    "num_samples": 10,
                    "seed": 0,
                    "batchsize_per_replica": 5,
                    "use_shuffle": False,
                    "transforms": [
                        {
                            "name": "apply_transform_to_key",
                            "transforms": [
                                {"name": "ToTensor"},
                                {
                                    "name": "Normalize",
                                    "mean": [0.485, 0.456, 0.406],
                                    "std": [0.229, 0.224, 0.225],
                                },
                            ],
                            "key": "input",
                        }
                    ],
                },
            },
            "model": {
                "name": "mlp",
                # 3x20x20 = 1200
                "input_dim": 1200,
                "output_dim": 1000,
                "hidden_dims": [10],
            },
            "meters": {"accuracy": {"topk": [1]}},
            "optimizer": self._get_optimizer_config(skip_param_schedulers),
        }

    def _build_task(self, num_epochs, skip_param_schedulers=False):
        config = self._get_config(skip_param_schedulers)
        config["optimizer"]["num_epochs"] = num_epochs
        task = (
            ClassificationTask()
            .set_num_epochs(num_epochs)
            .set_loss(build_loss(config["loss"]))
            .set_model(build_model(config["model"]))
            .set_optimizer(build_optimizer(config["optimizer"]))
        )
        for phase_type in ["train", "test"]:
            dataset = build_dataset(config["dataset"][phase_type])
            task.set_dataset(dataset, phase_type)

        self.assertTrue(task is not None)
        return task

    def test_param_scheduler_epoch(self):
        task = self._build_task(num_epochs=3)

        where_list = []

        def scheduler_mock(where):
            where_list.append(where)
            return 0.1

        mock = Mock(side_effect=scheduler_mock)
        mock.update_interval = UpdateInterval.EPOCH
        task.optimizer.param_schedulers["lr"] = mock

        trainer = LocalTrainer()
        trainer.train(task)

        self.assertEqual(where_list, [0, 1 / 3, 2 / 3])

    def test_param_scheduler_step(self):
        task = self._build_task(num_epochs=3)

        where_list = []

        def scheduler_mock(where):
            where_list.append(where)
            return 0.1

        mock = Mock(side_effect=scheduler_mock)
        mock.update_interval = UpdateInterval.STEP
        task.optimizer.param_schedulers["lr"] = mock

        trainer = LocalTrainer()
        trainer.train(task)

        # We have 10 samples, batch size is 5. Each epoch is done in two steps.
        self.assertEqual(where_list, [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])

    def test_no_param_schedulers(self):
        task = self._build_task(num_epochs=3, skip_param_schedulers=True)

        # there should be no param schedulers
        self.assertEqual(task.optimizer.param_schedulers, {})

        # we should still be able to train the task
        trainer = LocalTrainer()
        trainer.train(task)

    def test_hook(self):
        task = self._build_task(num_epochs=3)

        lr_list = []
        weight_decay_list = []
        momentum_list = []

        test_instance = self

        class TestHook(ClassyHook):
            on_rendezvous = ClassyHook._noop
            on_start = ClassyHook._noop
            on_phase_start = ClassyHook._noop
            on_sample = ClassyHook._noop
            on_forward = ClassyHook._noop
            on_loss_and_meter = ClassyHook._noop
            on_backward = ClassyHook._noop
            on_phase_end = ClassyHook._noop
            on_end = ClassyHook._noop

            def on_step(self, task: ClassyTask, local_variables) -> None:
                if not task.train:
                    return

                # make sure we have non-zero param groups
                test_instance.assertGreater(
                    len(task.optimizer.optimizer.param_groups), 0
                )
                # test that our overrides work on the underlying PyTorch optimizer
                for param_group in task.optimizer.optimizer.param_groups:
                    test_instance.assertEqual(
                        param_group["lr"], task.optimizer.parameters.lr
                    )
                    test_instance.assertEqual(
                        param_group["weight_decay"],
                        task.optimizer.parameters.weight_decay,
                    )
                    test_instance.assertEqual(
                        param_group["momentum"], task.optimizer.parameters.momentum
                    )
                lr_list.append(param_group["lr"])
                weight_decay_list.append(param_group["weight_decay"])
                momentum_list.append(param_group["momentum"])

        task.set_hooks([TestHook()])

        def scheduler_mock(where):
            return where

        mock = Mock(side_effect=scheduler_mock)
        mock.update_interval = UpdateInterval.STEP
        task.optimizer.param_schedulers["lr"] = mock

        trainer = LocalTrainer()
        trainer.train(task)

        # We have 10 samples, batch size is 5. Each epoch takes two steps. So,
        # there will be a total of 6 steps.
        # the lr scheduler uses a step update interval
        self.assertEqual(lr_list, [0 / 6, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
        # the weight decay scheduler uses an epoch update interval
        self.assertEqual(weight_decay_list, [0 / 6, 0 / 6, 4 / 6, 4 / 6, 8 / 6, 8 / 6])
        self.assertEqual(momentum_list, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
