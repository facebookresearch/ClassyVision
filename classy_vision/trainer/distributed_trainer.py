#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Optional

import torch
from classy_vision.generic.distributed_util import (
    get_rank,
    get_world_size,
    set_cpu_device,
    set_cuda_device_index,
)

from .classy_trainer import ClassyTrainer


def _init_env_vars():
    """Function sets up default environment variables for distributed training.
    """
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"

    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"


def _init_distributed(use_gpu: bool):
    """Function perform distributed setup for DDP.

    Requires the script to be started with torch.distributed.launch
    script and uses environment variables for node finding.

    Args:
        use_gpu: If true, use distributed GPU training, else use CPU
    """
    distributed_world_size = int(os.environ["WORLD_SIZE"])
    distributed_rank = int(os.environ["RANK"])
    backend = "nccl" if use_gpu else "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=distributed_world_size,
        rank=distributed_rank,
    )


class DistributedTrainer(ClassyTrainer):
    """Distributed trainer for using multiple training processes
    """

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        num_dataloader_workers: int = 0,
        dataloader_mp_context: Optional[str] = None,
    ):
        """Constructor for DistributedTrainer.

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
        _init_env_vars()
        _init_distributed(self.use_gpu)
        logging.info(
            f"Done setting up distributed process_group with rank {get_rank()}"
            + f", world_size {get_world_size()}"
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        if self.use_gpu:
            logging.info("Using GPU, CUDA device index: {}".format(local_rank))
            set_cuda_device_index(local_rank)
        else:
            logging.info("Using CPU")
            set_cpu_device()
