#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import shutil
import tempfile
from test.generic.config_utils import get_test_task_config
from test.generic.hook_test_utils import HookTestBase

import torch
from classy_vision.hooks import TorchscriptHook
from classy_vision.models import ResNet
from classy_vision.tasks import build_task


TORCHSCRIPT_FILE = "torchscript.pt"


class TestTorchscriptHook(HookTestBase):
    def setUp(self) -> None:
        self.base_dir = tempfile.mkdtemp()
        self.orig_wrapper_cls = ResNet.wrapper_cls

    def tearDown(self) -> None:
        shutil.rmtree(self.base_dir)
        ResNet.wrapper_cls = self.orig_wrapper_cls

    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        config = {"torchscript_folder": "/test/"}
        invalid_config = copy.deepcopy(config)
        invalid_config["torchscript_folder"] = 12

        self.constructor_test_helper(
            config=config,
            hook_type=TorchscriptHook,
            hook_registry_name="torchscript",
            invalid_configs=[invalid_config],
        )

    def execute_hook(self, config, torchscript_folder, torchscript_hook) -> None:
        task = build_task(config)
        task.prepare()

        # create checkpoint dir, verify on_start hook runs
        os.mkdir(torchscript_folder)
        torchscript_hook.on_start(task)

        task.train = True
        # call the on end function
        torchscript_hook.on_end(task)

        # load torchscript file
        torchscript_file_name = (
            f"{torchscript_hook.torchscript_folder}/{TORCHSCRIPT_FILE}"
        )
        torchscript = torch.jit.load(torchscript_file_name)
        # compare model load from checkpoint vs torchscript
        with torch.no_grad():
            batchsize = 1
            model = task.model
            input_data = torch.randn(
                (batchsize,) + model.input_shape, dtype=torch.float
            )
            if torch.cuda.is_available():
                input_data = input_data.cuda()
                model = model.cuda()
                torchscript = torchscript.cuda()
            checkpoint_out = model(input_data)
            torchscript_out = torchscript(input_data)
            self.assertTrue(torch.allclose(checkpoint_out, torchscript_out, atol=1e-5))

    def test_torchscripting_using_trace(self):
        """
        Test that the save_torchscript function works as expected with trace
        """
        config = get_test_task_config()
        torchscript_folder = self.base_dir + "/torchscript_end_test/"

        # create a torchscript hook using trace
        torchscript_hook = TorchscriptHook(torchscript_folder)
        self.execute_hook(config, torchscript_folder, torchscript_hook)

    def test_torchscripting_using_script(self):
        """
        Test that the save_torchscript function works as expected with script
        """
        config = get_test_task_config()
        # Setting wrapper_cls to None to make ResNet model torchscriptable
        ResNet.wrapper_cls = None
        torchscript_folder = self.base_dir + "/torchscript_end_test/"

        # create a torchscript hook using script
        torchscript_hook = TorchscriptHook(torchscript_folder, use_trace=False)
        self.execute_hook(config, torchscript_folder, torchscript_hook)
