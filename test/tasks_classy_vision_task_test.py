#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_args, get_test_task_config

from classy_vision.criterions import build_criterion
from classy_vision.dataset import build_dataset
from classy_vision.models import build_model
from classy_vision.optim import build_optimizer
from classy_vision.tasks import build_task
from classy_vision.tasks.classy_task import ClassyTask


class TestClassyTask(unittest.TestCase):
    def test_build_task(self):
        config = get_test_task_config()
        args = get_test_args()
        task = build_task(config, args)
        self.assertTrue(isinstance(task, ClassyTask))

    def test_get_state(self):
        config = get_test_task_config()
        criterion = build_criterion(config["criterion"])
        task = (
            ClassyTask(num_phases=1)
            .set_criterion(criterion)
            .set_model(build_model(config["model"]))
            .set_optimizer(build_optimizer(config["optimizer"]))
        )
        for split in ["train", "test"]:
            dataset = build_dataset(config["dataset"][split])
            task.set_dataset(dataset, split)

        task.prepare(num_workers=1, pin_memory=False)

        args = get_test_args()
        task = build_task(config, args)
        task.prepare(num_workers=1, pin_memory=False)
