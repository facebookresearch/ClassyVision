#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from classy_vision.generic.distributed_util import set_cpu_device, set_cuda_device_index

from .classy_trainer import ClassyTrainer


class LocalTrainer(ClassyTrainer):
    """Trainer to be used if you want want use only a single training process.
    """

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        num_dataloader_workers: int = 0,
        dataloader_mp_context: Optional[str] = None,
    ):
        """Constructor for LocalTrainer.

        Args:
            use_gpu: If true, then use GPU 0 for training.
                If None, then check if we have GPUs available, if we do
                then use GPU for training.
            num_dataloader_workers: Number of CPU processes doing dataloading
                per GPU. If 0, then dataloading is done on main thread.
            dataloader_mp_context: Determines how to launch
                new processes for dataloading. Must be one of "fork", "forkserver",
                "spawn". If None, process launching is inherited from parent.
        """
        super().__init__(
            use_gpu=use_gpu,
            num_dataloader_workers=num_dataloader_workers,
            dataloader_mp_context=dataloader_mp_context,
        )
        if self.use_gpu:
            logging.info("Using GPU, CUDA device index: {}".format(0))
            set_cuda_device_index(0)
        else:
            logging.info("Using CPU")
            set_cpu_device()
