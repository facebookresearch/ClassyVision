#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from test.generic.config_utils import (
    get_distributed_launch_cmd,
    get_fast_test_task_config,
)
from test.generic.optim_test_util import TestOptimizer

import classy_vision.optim  # NOQA
import torch
import torch.distributed as dist
from classy_vision.optim.zero import ZeRO


def dist_init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=dist.Backend.GLOO, rank=rank, world_size=world_size)


class TestOptimizerStateShardingIntegration(unittest.TestCase, TestOptimizer):
    @staticmethod
    def _maybe_destro_dist():
        if dist.is_initialized():
            logging.debug("Destroy previous torch dist process group")
            dist.destroy_process_group()

    def setUp(self):
        self._maybe_destro_dist()
        dist_init(0, 1)

    def tearDown(self):
        self._maybe_destro_dist()

    def _get_config(self):
        return {"name": "zero", "base_optimizer": {"name": "sgd"}, "num_epochs": 3}

    def _instance_to_test(self):
        return ZeRO


class TestOptimizerStateSharding(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).parent.absolute()

        # Save the task config file on disk
        config = self._get_task_config()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file_io:
            json.dump(config, file_io)
            file_io.flush()
            self.config_path = file_io.name

    def tearDown(self):
        if self.config_path is not None:
            os.unlink(self.config_path)

    def _get_task_config(self):
        config = get_fast_test_task_config()
        config["optimizer"] = {
            "name": "zero",
            "base_optimizer": {"name": "sgd", "momentum": 0.9},
        }

        return config

    @unittest.skipUnless(torch.cuda.is_available(), "This test needs a gpu to run")
    def test_oss_sgd(self):
        """
        Test that the optimizer is correctly instantiated and that a task can run
        """
        num_processes = 2

        cmd = get_distributed_launch_cmd(
            num_processes=num_processes,
            trainer_path=f"{Path(__file__).parent.absolute()}/../classy_train.py",
            config_path=self.config_path,
        )

        result = subprocess.run(cmd, shell=True)
        self.assertEqual(result.returncode, 0)
