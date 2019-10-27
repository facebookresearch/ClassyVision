#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_task_config

from classy_vision.dataset import build_dataset
from classy_vision.losses import build_loss
from classy_vision.models import build_model
from classy_vision.optim import build_optimizer
from classy_vision.tasks import ClassificationTask, build_task


class TestClassificationTask(unittest.TestCase):
    def test_build_task(self):
        config = get_test_task_config()
        task = build_task(config)
        self.assertTrue(isinstance(task, ClassificationTask))

    def test_get_state(self):
        config = get_test_task_config()
        loss = build_loss(config["loss"])
        task = (
            ClassificationTask()
            .set_num_epochs(1)
            .set_loss(loss)
            .set_model(build_model(config["model"]))
            .set_optimizer(build_optimizer(config["optimizer"]))
        )
        for split in ["train", "test"]:
            dataset = build_dataset(config["dataset"][split])
            task.set_dataset(dataset, split)

        task.prepare(num_dataloader_workers=1, pin_memory=False)

        task = build_task(config)
        task.prepare(num_dataloader_workers=1, pin_memory=False)
