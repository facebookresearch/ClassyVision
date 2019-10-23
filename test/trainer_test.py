#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from classy_vision.dataset import build_dataset
from classy_vision.hooks import LossLrMeterLoggingHook
from classy_vision.losses import build_loss
from classy_vision.meters.accuracy_meter import AccuracyMeter
from classy_vision.models import build_model
from classy_vision.optim import build_optimizer
from classy_vision.tasks import ClassificationTask
from classy_vision.trainer import ClassyTrainer


class TestClassyTrainer(unittest.TestCase):
    def _get_config(self):
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
        task = (
            ClassificationTask()
            .set_num_epochs(10)
            .set_loss(build_loss(config["loss"]))
            .set_model(build_model(config["model"]))
            .set_optimizer(build_optimizer(config["optimizer"]))
            .set_meters([AccuracyMeter(topk=[1])])
            .set_hooks([LossLrMeterLoggingHook()])
        )
        for split in ["train", "test"]:
            dataset = build_dataset(config["dataset"][split])
            task.set_dataset(dataset, split)

        self.assertTrue(task is not None)

        trainer = ClassyTrainer(use_gpu=False)
        trainer.train(task)
        accuracy = task.meters[0].value["top_1"]
        self.assertAlmostEqual(accuracy, 1.0)
