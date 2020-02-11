#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from classy_vision.generic.distributed_util import barrier
from classy_vision.tasks import ClassyTask


class ClassyTrainer:
    """Base class for shared training code.

    A trainer is responsible for setting up the environment for
    training, for instance: configuring rendezvous for distributed
    training, deciding what GPU to use and so on. Trainers also
    control the outer portion of the training loop, but delegate to
    the task to decide how exactly to perform inference, compute loss
    etc. That allows combining tasks with different trainers depending
    on whether you want to train on your current machine, AWS cluster
    etc.

    """

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        num_dataloader_workers: int = 0,
        dataloader_mp_context: Optional[str] = None,
    ):
        """Constructor for ClassyTrainer.

        Args:
            use_gpu: If true, then use GPUs for training.
                If None, then check if we have GPUs available, if we do
                then use GPU for training.
            num_dataloader_workers: Number of CPU processes doing dataloading
                per GPU. If 0, then dataloading is done on main thread.
            dataloader_mp_context: Determines how to launch
                new processes for dataloading. Must be one of "fork", "forkserver",
                "spawn". If None, process launching is inherited from parent.
        """
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu
        self.num_dataloader_workers = num_dataloader_workers
        self.dataloader_mp_context = dataloader_mp_context

    def train(self, task: ClassyTask):
        """Runs training phases, phases are generated from the config.

        Args:
            task: Task to be used in training. It should contain
                everything that is needed for training
        """

        pin_memory = self.use_gpu and torch.cuda.device_count() > 1
        task.prepare(
            num_dataloader_workers=self.num_dataloader_workers,
            pin_memory=pin_memory,
            use_gpu=self.use_gpu,
            dataloader_mp_context=self.dataloader_mp_context,
        )
        assert isinstance(task, ClassyTask)

        # make sure all the workers start training at the same time
        # this helps catch hangs which would have happened elsewhere
        barrier()

        local_variables = {}

        task.on_start(local_variables)
        while not task.done_training():
            task.on_phase_start(local_variables)
            while True:
                try:
                    task.step(self.use_gpu, local_variables)
                except StopIteration:
                    break
            task.on_phase_end(local_variables)
        task.on_end(local_variables)
