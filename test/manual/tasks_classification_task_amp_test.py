#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from test.generic.config_utils import get_fast_test_task_config, get_test_task_config

import torch
from classy_vision.tasks import ClassificationTask, build_task
from classy_vision.trainer import LocalTrainer


class TestClassificationTaskAMP(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "This test needs a gpu to run")
    def test_build_task(self):
        config = get_test_task_config()
        task = build_task(config)
        self.assertTrue(isinstance(task, ClassificationTask))
        # check that AMP is disabled by default
        self.assertIsNone(task.amp_args)

        # test a valid APEX AMP opt level
        config = copy.deepcopy(config)
        config["amp_args"] = {"opt_level": "O1"}
        task = build_task(config)
        self.assertTrue(isinstance(task, ClassificationTask))

        # test a valid Pytorch AMP
        config = copy.deepcopy(config)
        config["amp_args"] = {"amp_type": "pytorch"}
        task = build_task(config)
        self.assertTrue(isinstance(task, ClassificationTask))

    @unittest.skipUnless(torch.cuda.is_available(), "This test needs a gpu to run")
    def test_training(self):
        # Test an Apex AMP training
        config = get_fast_test_task_config()
        config["amp_args"] = {"opt_level": "O2"}
        task = build_task(config)
        task.set_use_gpu(True)
        trainer = LocalTrainer()
        trainer.train(task)

        # Test a Pytorch AMP training
        config["amp_args"] = {"amp_type": "pytorch"}
        task = build_task(config)
        task.set_use_gpu(True)
        trainer = LocalTrainer()
        trainer.train(task)
