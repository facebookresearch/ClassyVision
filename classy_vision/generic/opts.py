#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from classy_vision.generic.util import is_pos_int


def add_generic_args(parser):
    """
    Adds generic command-line arguments for convnet training / testing to parser.
    """
    parser.add_argument(
        "--config_file", type=str, help="path to config file for model", required=True
    )
    parser.add_argument(
        "--checkpoint_folder",
        default="",
        type=str,
        help="""folder to use for saving checkpoints:
                        epochal checkpoints are stored as model_<epoch>.torch,
                        latest epoch checkpoint is at checkpoint.torch""",
    )
    parser.add_argument(
        "--checkpoint_load_path",
        default="",
        type=str,
        help="""path to load a checkpoint from, which can be a file or a directory:
                        If the path is a directory, the checkpoint file is assumed to be
                        checkpoint.torch""",
    )
    parser.add_argument(
        "--pretrained_checkpoint_path",
        default="",
        type=str,
        help="""path to load a pre-trained checkpoints from, which can be a file or a
                        directory:
                        If the path is a directory, the checkpoint file is assumed to be
                        checkpoint.torch. This checkpoint is only used for fine-tuning
                        tasks, and training will not resume from this checkpoint.""",
    )
    parser.add_argument(
        "--checkpoint_period",
        default=1,
        type=int,
        help="""Checkpoint every x phases (default 1)""",
    )
    parser.add_argument(
        "--show_progress",
        default=False,
        action="store_true",
        help="shows progress bar during training / testing",
    )
    parser.add_argument(
        "--skip_tensorboard",
        default=False,
        action="store_true",
        help="do not perform tensorboard visualization",
    )
    parser.add_argument(
        "--visdom_server",
        default="",
        type=str,
        help="visdom server to use (default None)",
    )
    parser.add_argument(
        "--visdom_port",
        default=8097,
        type=int,
        help="port of visdom server (default = 8097)",
    )
    parser.add_argument(
        "--profiler",
        default=False,
        action="store_true",
        help="specify this argument to profile training code",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="specify this argument for debugging mode",
    )
    parser.add_argument(
        "--ignore_checkpoint_config",
        default=False,
        action="store_true",
        help="""specify this argument to ignore
                        the compatibility of the config (or lack of config) attached
                        to the checkpoint; this will allow mismatches between
                        the training specified in the config and the
                        actual training of the model""",
    )
    parser.add_argument(
        "--log_freq",
        default=5,
        type=int,
        help="Logging frequency for LossLrMeterLoggingHook (default 5)",
    )
    parser.add_argument(
        "--image_backend",
        default="PIL",
        type=str,
        help="torchvision image decoder backend (PIL or accimage). Default PIL",
    )
    parser.add_argument(
        "--video_backend",
        default="pyav",
        type=str,
        help="torchvision video decoder backend (pyav or video_reader). Default pyav",
    )
    parser.add_argument(
        "--distributed_backend",
        default="none",
        type=str,
        help="""Distributed backend: either 'none' (for non-distributed runs)
             or 'ddp' (for distributed runs). Default none.""",
    )

    return parser


def check_generic_args(args):
    """
    Perform assertions on generic command-line arguments.
    """

    # check types and values:
    assert is_pos_int(args.visdom_port), "incorrect visdom port"

    # create checkpoint folder if it does not exist:
    if args.checkpoint_folder != "" and not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder, exist_ok=True)
        assert os.path.exists(args.checkpoint_folder), (
            "could not create folder %s" % args.checkpoint_folder
        )

    # when in debugging mode, enter debugger upon error:
    if args.debug:
        import sys

        from classy_vision.generic.debug import debug_info

        sys.excepthook = debug_info

    # check visdom server name:
    if args.visdom_server != "":
        if args.visdom_server.startswith("https://"):
            print("WARNING: Visdom does not work over HTTPS.")
            args.visdom_server = args.visdom_server[8:]
        if not args.visdom_server.startswith("http://"):
            args.visdom_server = "http://%s" % args.visdom_server

    # return input arguments:
    return args


def get_parser():
    """
    Return a standard command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="""Start a Classy Vision training job.

    This can be used for training on your local machine, using CPU or GPU, and
    for distributed training. This script also supports Tensorboard, Visdom and
    checkpointing."""
    )

    parser = add_generic_args(parser)
    return parser


def parse_train_arguments(parser=None):
    """
    Assert and parse the command-line arguments of a given (or default) parser.
    """

    # set input arguments:
    if parser is None:
        parser = get_parser()

    # parse input arguments:
    args = parser.parse_args()

    # assertions:
    args = check_generic_args(args)
    return args
