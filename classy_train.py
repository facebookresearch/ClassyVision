#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
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
from classy_vision.tasks import build_task
from classy_vision.trainer import DistributedTrainer


# run all the things:
def main(args):
    torch.manual_seed(0)

    # Loads config, sets up task
    config = load_json(args.config)

    task = build_task(config, args)

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

    trainer = DistributedTrainer(
        hooks, args.device == "gpu", num_workers=args.num_workers
    )
    trainer.train(task)


# run all the things:
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info("Generic convolutional network trainer.")
    args = parse_train_arguments()
    main(args)
