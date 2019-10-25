#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from classy_vision.generic.distributed_util import barrier, is_distributed_training_run
from classy_vision.hooks import ClassyHookFunctions
from classy_vision.tasks import ClassyTask


class ClassyTrainer:
    def __init__(self, use_gpu=None, num_workers=0):
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu
        self.num_workers = num_workers

    def train(self, task: ClassyTask):
        """
        Runs training phases, phases are generated from the config.
        """

        pin_memory = self.use_gpu and torch.cuda.device_count() > 1
        task.prepare(
            num_workers=self.num_workers, pin_memory=pin_memory, use_gpu=self.use_gpu
        )
        assert isinstance(task, ClassyTask)

        if is_distributed_training_run():
            task.init_distributed_data_parallel_model()

        local_variables = {}
        task.run_hooks(local_variables, ClassyHookFunctions.on_start.name)

        while not task.done_training():
            task.advance_phase()

            # Start phase hooks
            task.run_hooks(local_variables, ClassyHookFunctions.on_phase_start.name)
            while True:
                # Process next sample
                try:
                    task.train_step(self.use_gpu, local_variables)
                except StopIteration:
                    break

            logging.info("Syncing meters on phase end...")
            for meter in task.meters:
                meter.sync_state()
            logging.info("...meters synced")
            barrier()
            task.run_hooks(local_variables, ClassyHookFunctions.on_phase_end.name)

        task.run_hooks(local_variables, ClassyHookFunctions.on_end.name)
