#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from classy_vision.criterions import build_criterion
from classy_vision.tasks.classy_vision_task import ClassyVisionTask
from classy_vision.trainer import ClassyTrainer


class TestClassyTrainer(unittest.TestCase):
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
                    "batchsize_per_replica": 3,
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
                    "batchsize_per_replica": 1,
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

    def test_cpu_training(self):
        """Checks we can train a small MLP model on a CPU."""
        config = self._get_config()
        task = ClassyVisionTask(
            num_phases=10,
            dataset_config=config["dataset"],
            model_config=config["model"],
            optimizer_config=config["optimizer"],
            meter_config=config["meters"],
        ).set_criterion(build_criterion(config["criterion"]))
        self.assertTrue(task is not None)

        trainer = ClassyTrainer(hooks=[], use_gpu=False)
        trainer.train(task)
        accuracy = task.meters[0].value["top_1"]
        self.assertAlmostEqual(accuracy, 1.0)
