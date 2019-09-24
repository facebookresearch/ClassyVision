#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
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
from classy_vision.generic.opts import parse_train_arguments
from classy_vision.generic.util import load_checkpoint, load_json
from classy_vision.hooks import (
    CheckpointHook,
    LossLrMeterLoggingHook,
    ModelComplexityHook,
    ModelTensorboardHook,
    ProfilerHook,
    ProgressBarHook,
    TimeMetricsHook,
    VisdomHook,
)
from classy_vision.tasks import setup_task
from classy_vision.trainer import ClassyTrainer


def _init_env_vars():
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"

    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"


def _init_distributed(device):
    """ perform distributed setup, requires the script to be started with
        torch.distributed.launch script.
    """
    distributed_world_size = int(os.environ["WORLD_SIZE"])
    distributed_rank = int(os.environ["RANK"])
    backend = "nccl" if device == "gpu" else "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=distributed_world_size,
        rank=distributed_rank,
    )


# run all the things:
def main(args):
    torch.manual_seed(0)

    # Loads config, sets up task
    config = load_json(args.config)

    _init_env_vars()
    _init_distributed(args.device)

    logging.info(
        "Done setting up distributed process_group with rank {}, world_size {}".format(
            get_rank(), get_world_size()
        )
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.device == "gpu":
        logging.info("Using GPU, CUDA device index: {}".format(local_rank))
        set_cuda_device_index(local_rank)
    else:
        logging.info("Using CPU")
        set_cpu_device()
    task = setup_task(config, args)

    # Load checkpoint, if available
    checkpoint = load_checkpoint(args.checkpoint_folder, args.device)

    task.set_checkpoint(checkpoint)

    hooks = [LossLrMeterLoggingHook(), ModelComplexityHook(), TimeMetricsHook()]
    if not args.skip_tensorboard:
        hooks.append(ModelTensorboardHook())
    if args.checkpoint_folder != "":
        hooks.append(
            CheckpointHook(
                args.checkpoint_folder, args, checkpoint_period=args.checkpoint_period
            )
        )
    if args.profiler:
        hooks.append(ProfilerHook())
    if args.show_progress:
        hooks.append(ProgressBarHook())
    if args.visdom_server != "":
        hooks.append(VisdomHook(args.visdom_server, args.visdom_port))

    trainer = ClassyTrainer(hooks, args.device == "gpu")
    trainer.train(task)


# run all the things:
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info("Generic convolutional network trainer.")
    args = parse_train_arguments()
    main(args)
