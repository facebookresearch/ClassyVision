#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_args, get_test_task_config

from classy_vision.criterions import build_criterion
from classy_vision.models import build_model
from classy_vision.optim import build_optimizer
from classy_vision.tasks import build_task
from classy_vision.tasks.classy_vision_task import ClassyVisionTask


class TestClassyVisionTask(unittest.TestCase):
    def test_build_task(self):
        config = get_test_task_config()
        args = get_test_args()
        task = build_task(config, args)
        self.assertTrue(isinstance(task, ClassyVisionTask))

    def test_get_state(self):
        config = get_test_task_config()
        model = build_model(config["model"])
        criterion = build_criterion(config["criterion"])
        task = (
            ClassyVisionTask(num_phases=1, dataset_config=config["dataset"])
            .set_criterion(criterion)
            .set_model(model)
            .set_optimizer(build_optimizer(config["optimizer"], model))
        )

        state = task.build_initial_state(num_workers=1, pin_memory=False)
        self.assertTrue(state is not None)

        args = get_test_args()
        task = build_task(config, args)
        state = task.build_initial_state(num_workers=1, pin_memory=False)
