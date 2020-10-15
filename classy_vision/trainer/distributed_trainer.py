#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from classy_vision.generic.distributed_util import (
    get_rank,
    get_world_size,
    set_cpu_device,
    set_cuda_device_index,
)

from .classy_trainer import ClassyTrainer


def _init_env_vars(use_gpu: bool):
    """Function sets up default environment variables for distributed training.

    Args:
        use_gpu: If true, set NCCL environment for GPUs
    """
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"

    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    if use_gpu:
        # From https://github.com/pytorch/elastic/blob/4175e9ec3ac346b89dab13eeca00e8f00b6daa8f/examples/imagenet/main.py#L156  # noqa B950
        # when using NCCL, on failures, surviving nodes will deadlock on NCCL ops
        # because NCCL uses a spin-lock on the device. Set this env var to enable a
        # watchdog thread that will destroy stale NCCL communicators
        os.environ["NCCL_BLOCKING_WAIT"] = "1"


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
    """Distributed trainer for using multiple training processes"""

    def train(self, task):
        _init_env_vars(task.use_gpu)
        _init_distributed(task.use_gpu)
        logging.info(
            f"Done setting up distributed process_group with rank {get_rank()}"
            + f", world_size {get_world_size()}"
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        if task.use_gpu:
            logging.info("Using GPU, CUDA device index: {}".format(local_rank))
            set_cuda_device_index(local_rank)
        else:
            logging.info("Using CPU")
            set_cpu_device()

        super().train(task)
