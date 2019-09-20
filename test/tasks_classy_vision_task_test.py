#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_args, get_test_task_config

from classy_vision.tasks import setup_task
from classy_vision.tasks.classy_vision_task import ClassyVisionTask


class TestClassyVisionTask(unittest.TestCase):
    def test_setup_task(self):
        config = get_test_task_config()
        args = get_test_args()
        task = setup_task(config, args, local_rank=0)
        self.assertTrue(isinstance(task, ClassyVisionTask))

    def test_get_state(self):
        config = get_test_task_config()
        task = ClassyVisionTask(
            device_type="cpu",
            num_phases=1,
            dataset_config=config["dataset"],
            model_config=config["model"],
            optimizer_config=config["optimizer"],
            criterion_config=config["criterion"],
            meter_config={},
            test_only=False,
            num_workers=1,
            pin_memory=False,
        )

        state = task.build_initial_state()
        self.assertTrue(state is not None)

        args = get_test_args()
        task = setup_task(config, args, local_rank=0)
        state = task.build_initial_state()
