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
    TensorboardPlotHook,
    TimeMetricsHook,
    VisdomHook,
)
from classy_vision.tasks import FineTuningTask, build_task
from classy_vision.trainer import DistributedTrainer
from torchvision import set_video_backend


def main(args):
    # Global settings
    torch.manual_seed(0)
    set_video_backend(args.video_backend)

    # Loads config, sets up task
    config = load_json(args.config)

    task = build_task(config)

    # Load checkpoint, if available
    checkpoint = load_checkpoint(args.checkpoint_folder, args.device)
    task.set_checkpoint(checkpoint)

    pretrained_checkpoint = load_checkpoint(
        args.pretrained_checkpoint_folder, args.device
    )
    if pretrained_checkpoint is not None:
        assert isinstance(
            task, FineTuningTask
        ), "Can only use a pretrained checkpoint for fine tuning tasks"
        task.set_pretrained_checkpoint(pretrained_checkpoint)

    hooks = [
        LossLrMeterLoggingHook(args.log_freq),
        ModelComplexityHook(),
        TimeMetricsHook(),
    ]
    if not args.skip_tensorboard:
        try:
            from tensorboardX import SummaryWriter

            tb_writer = SummaryWriter(log_dir="/tmp/tensorboard")
            hooks.append(TensorboardPlotHook(tb_writer))
            hooks.append(ModelTensorboardHook(tb_writer))
        except ImportError:
            logging.warning("tensorboardX not installed, skipping tensorboard hooks")
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

    task.set_hooks(hooks)

    trainer = DistributedTrainer(
        args.device == "gpu", num_dataloader_workers=args.num_workers
    )
    trainer.train(task)


# run all the things:
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info("Generic convolutional network trainer.")
    args = parse_train_arguments()
    main(args)
