#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is the main script used for training Classy Vision jobs.

This can be used for training on your local machine, using CPU or GPU, and
for distributed training. This script also supports Tensorboard, Visdom and
checkpointing.

Example:
    For training locally, simply specify a configuration file and whether
    to use CPU or GPU:

        $ ./classy_train.py --device gpu --config configs/my_config.json

    For distributed training, this can be invoked via
    :func:`torch.distributed.launch`. For instance

        $ python -m torch.distributed.launch \
                 --nnodes=1 \
                 --nproc_per_node=1 \
                 --master_addr=localhost \
                 --master_port=29500 \
                 --use_env \
                 classy_train.py \
                 --device=gpu \
                 --config=configs/resnet50_synthetic_image_classy_config.json \
                 --num_workers=1 \
                 --log_freq=100

    For other use cases, try

        $ ./classy_train.py --help
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from classy_vision.generic.opts import check_generic_args, parse_train_arguments
from classy_vision.generic.registry_utils import import_all_packages_from_directory
from classy_vision.generic.util import load_checkpoint, load_json
from classy_vision.hooks import (
    CheckpointHook,
    LossLrMeterLoggingHook,
    ProfilerHook,
    ProgressBarHook,
    TensorboardPlotHook,
    VisdomHook,
)
from classy_vision.tasks import FineTuningTask, build_task
from classy_vision.trainer import DistributedTrainer, LocalTrainer
from torchvision import set_image_backend, set_video_backend


try:
    import hydra

    hydra_available = True
except ImportError:
    hydra_available = False


def main(args, config):
    # Global flags
    torch.manual_seed(0)
    set_image_backend(args.image_backend)
    set_video_backend(args.video_backend)

    task = build_task(config)

    # Load checkpoint, if available.
    checkpoint = load_checkpoint(args.checkpoint_load_path)
    task.set_checkpoint(checkpoint)

    # Load a checkpoint contraining a pre-trained model. This is how we
    # implement fine-tuning of existing models.
    pretrained_checkpoint = load_checkpoint(args.pretrained_checkpoint_path)
    if pretrained_checkpoint is not None:
        assert isinstance(
            task, FineTuningTask
        ), "Can only use a pretrained checkpoint for fine tuning tasks"
        task.set_pretrained_checkpoint(pretrained_checkpoint)

    # Configure hooks to do tensorboard logging, checkpoints and so on
    task.set_hooks(configure_hooks(args, config))

    use_gpu = None
    if args.device is not None:
        use_gpu = args.device == "gpu"
        assert torch.cuda.is_available() or not use_gpu, "CUDA is unavailable"

    # LocalTrainer is used for a single node. DistributedTrainer will setup
    # training to use PyTorch's DistributedDataParallel.
    trainer_class = {"none": LocalTrainer, "ddp": DistributedTrainer}[
        args.distributed_backend
    ]

    trainer = trainer_class(use_gpu=use_gpu, num_dataloader_workers=args.num_workers)

    # That's it! When this call returns, training is done.
    trainer.train(task)

    output_folder = Path(args.checkpoint_folder).resolve()
    logging.info("Training successful!")
    logging.info(f'Results of this training run are available at: "{output_folder}"')


def configure_hooks(args, config):
    hooks = [LossLrMeterLoggingHook(args.log_freq)]

    # Make a folder to store checkpoints and tensorboard logging outputs
    suffix = datetime.now().isoformat()
    base_folder = f"{Path(__file__).parent}/output_{suffix}"
    if args.checkpoint_folder == "":
        args.checkpoint_folder = base_folder + "/checkpoints"
        os.makedirs(args.checkpoint_folder, exist_ok=True)

    logging.info(f"Logging outputs to {base_folder}")
    logging.info(f"Logging checkpoints to {args.checkpoint_folder}")

    if not args.skip_tensorboard:
        try:
            from tensorboardX import SummaryWriter

            tb_writer = SummaryWriter(log_dir=Path(base_folder) / "tensorboard")
            hooks.append(TensorboardPlotHook(tb_writer))
        except ImportError:
            logging.warning("tensorboardX not installed, skipping tensorboard hooks")

    args_dict = vars(args)
    args_dict["config"] = config
    hooks.append(
        CheckpointHook(
            args.checkpoint_folder, args_dict, checkpoint_period=args.checkpoint_period
        )
    )

    if args.profiler:
        hooks.append(ProfilerHook())
    if args.show_progress:
        hooks.append(ProgressBarHook())
    if args.visdom_server != "":
        hooks.append(VisdomHook(args.visdom_server, args.visdom_port))

    return hooks


if hydra_available:

    @hydra.main(config_path="hydra_configs/args.yaml")
    def hydra_main(cfg):
        args = cfg
        check_generic_args(cfg)
        config = cfg.config.to_container()
        main(args, config)


# run all the things:
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info("Classy Vision's default training script.")

    # This imports all modules in the same directory as classy_train.py
    # Because of the way Classy Vision's registration decorators work,
    # importing a module has a side effect of registering it with Classy
    # Vision. This means you can give classy_train.py a config referencing your
    # custom module (e.g. my_dataset) and it'll actually know how to
    # instantiate it.
    file_root = Path(__file__).parent
    import_all_packages_from_directory(file_root)

    if hydra_available:
        hydra_main()
    else:
        args = parse_train_arguments()
        config = load_json(args.config_file)
        main(args, config)
