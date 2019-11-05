#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_mlp_task_config

from classy_vision.dataset import build_dataset
from classy_vision.hooks import LossLrMeterLoggingHook
from classy_vision.losses import build_loss
from classy_vision.meters import AccuracyMeter
from classy_vision.models import build_model
from classy_vision.optim import build_optimizer
from classy_vision.tasks import ClassificationTask
from classy_vision.trainer import LocalTrainer


class TestLocalTrainer(unittest.TestCase):
    def test_training(self):
        """Checks we can train a small MLP model."""
        config = get_test_mlp_task_config()
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

        trainer = LocalTrainer()
        trainer.train(task)
        accuracy = task.meters[0].value["top_1"]
        self.assertAlmostEqual(accuracy, 1.0)
