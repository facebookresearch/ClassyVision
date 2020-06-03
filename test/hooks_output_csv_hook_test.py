#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import shutil
import tempfile
from test.generic.config_utils import get_fast_test_task_config
from test.generic.hook_test_utils import HookTestBase

import torch
from classy_vision.hooks import OutputCSVHook
from classy_vision.tasks import build_task
from classy_vision.trainer import LocalTrainer


def parse_csv(file_path):
    """Parses the csv file and returns number of rows"""

    num_rows = 0
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for _ in reader:
            num_rows += 1

    return num_rows


class TestCSVHook(HookTestBase):
    def setUp(self) -> None:
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.base_dir)

    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        folder = f"{self.base_dir}/constructor_test/"
        os.makedirs(folder)

        self.constructor_test_helper(
            config={"folder": folder},
            hook_type=OutputCSVHook,
            hook_registry_name="output_csv",
            invalid_configs=[],
        )

    def test_train(self) -> None:
        for use_gpu in {False, torch.cuda.is_available()}:
            folder = f"{self.base_dir}/train_test/{use_gpu}"
            os.makedirs(folder)

            task = build_task(get_fast_test_task_config(head_num_classes=2))

            csv_hook = OutputCSVHook(folder)
            task.set_hooks([csv_hook])
            task.set_use_gpu(use_gpu)

            trainer = LocalTrainer()
            trainer.train(task)

            self.assertEqual(parse_csv(csv_hook.output_path), 10)
