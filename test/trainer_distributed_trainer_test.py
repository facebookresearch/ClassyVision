#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from test.generic.config_utils import get_test_mlp_task_config

import torch


class TestDistributedTrainer(unittest.TestCase):
    def setUp(self):
        config = get_test_mlp_task_config()
        invalid_config = copy.deepcopy(config)
        invalid_config["name"] = "invalid_task"
        sync_bn_config = copy.deepcopy(config)
        sync_bn_config["batch_norm_sync_mode"] = "pytorch"
        self.config_files = {}
        for config_key, config in [
            ("config", config),
            ("invalid_config", invalid_config),
            ("sync_bn_config", sync_bn_config),
        ]:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                json.dump(config, f)
                f.flush()
                self.config_files[config_key] = f.name
        self.path = Path(__file__).parent.absolute()

    def tearDown(self):
        for config_file in self.config_files.values():
            os.unlink(config_file)

    def test_training(self):
        """Checks we can train a small MLP model."""

        num_processes = 2

        for config_key, expected_success in [
            ("invalid_config", False),
            ("config", True),
        ]:
            cmd = f"""{sys.executable} -m torch.distributed.launch \
            --nnodes=1 \
            --nproc_per_node={num_processes} \
            --master_addr=localhost \
            --master_port=29500 \
            --use_env \
            {self.path}/../classy_train.py \
            --config={self.config_files[config_key]} \
            --log_freq=100 \
            --distributed_backend=ddp
            """
            result = subprocess.run(cmd, shell=True)
            success = result.returncode == 0
            self.assertEqual(success, expected_success)

    @unittest.skipUnless(torch.cuda.is_available(), "This test needs a gpu to run")
    def test_sync_batch_norm(self):
        """Test that sync batch norm training doesn't hang."""

        num_processes = 2

        cmd = f"""{sys.executable} -m torch.distributed.launch \
        --nnodes=1 \
        --nproc_per_node={num_processes} \
        --master_addr=localhost \
        --master_port=29500 \
        --use_env \
        {self.path}/../classy_train.py \
        --config={self.config_files["sync_bn_config"]} \
        --log_freq=100 \
        --distributed_backend=ddp
        """
        result = subprocess.run(cmd, shell=True)
        self.assertEqual(result.returncode, 0)
