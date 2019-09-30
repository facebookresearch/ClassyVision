#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import Mock

from classy_vision.criterions import build_criterion
from classy_vision.optim.param_scheduler import UpdateInterval
from classy_vision.tasks.classy_vision_task import ClassyVisionTask
from classy_vision.trainer import ClassyTrainer


class TestParamSchedulerIntegration(unittest.TestCase):
    def _get_config(self):
        return {
            "criterion": {"name": "CrossEntropyLoss"},
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
                    "use_augmentation": False,
                    "use_shuffle": True,
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
                    "use_augmentation": False,
                    "use_shuffle": False,
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
            "optimizer": {
                "name": "sgd",
                "num_epochs": 10,
                "lr": 0.1,
                "weight_decay": 1e-4,
                "weight_decay_batchnorm": 0.0,
                "momentum": 0.9,
            },
        }

    def _build_task(self, num_phases):
        config = self._get_config()
        config["optimizer"]["num_epochs"] = num_phases
        task = ClassyVisionTask(
            device_type="cpu",
            num_phases=num_phases,
            dataset_config=config["dataset"],
            model_config=config["model"],
            optimizer_config=config["optimizer"],
            meter_config=config["meters"],
            test_only=False,
        ).set_criterion(build_criterion(config["criterion"]))

        self.assertTrue(task is not None)
        return task

    def test_param_scheduler_epoch(self):
        task = self._build_task(num_phases=3)

        where_list = []

        def scheduler_mock(where):
            where_list.append(where)
            return 0.1

        mock = Mock(side_effect=scheduler_mock)
        mock.update_interval = UpdateInterval.EPOCH
        task.optimizer._lr_scheduler = mock

        trainer = ClassyTrainer(hooks=[], use_gpu=False)
        trainer.train(task)

        self.assertEqual(where_list, [0, 1 / 3, 2 / 3])

    def test_param_scheduler_step(self):
        task = self._build_task(num_phases=3)

        where_list = []

        def scheduler_mock(where):
            where_list.append(where)
            return 0.1

        mock = Mock(side_effect=scheduler_mock)
        mock.update_interval = UpdateInterval.STEP
        task.optimizer._lr_scheduler = mock

        trainer = ClassyTrainer(hooks=[], use_gpu=False)
        trainer.train(task)

        # We have 10 samples, batch size is 5. Each epoch is done in two steps.
        self.assertEqual(where_list, [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
