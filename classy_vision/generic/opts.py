#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import classy_vision.generic.visualize as visualize
import torch
import torchvision
from classy_vision.generic.util import is_pos_int


def add_generic_args(parser):
    """
    Adds generic command-line arguments for convnet training / testing to parser.
    """
    parser.add_argument(
        "--config", default="", type=str, help="path to config file for model"
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to use: cpu or gpu (default = gpu)",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="number of dataloading workers (default = 4)",
    )
    parser.add_argument(
        "--checkpoint_folder",
        default="",
        type=str,
        help="""folder to use for checkpoints:
                        epochal checkpoints are stored as model_<epoch>.torch,
                        latest epoch checkpoint is at checkpoint.torch""",
    )
    parser.add_argument(
        "--checkpoint_period",
        default=1,
        type=int,
        help="""Checkpoint every x phases (default 1)""",
    )
    parser.add_argument(
        "--test_only",
        default=False,
        action="store_true",
        help="do not perform training: only test model",
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

    return parser


def check_generic_args(args):
    """
    Perform assertions on generic command-line arguments.
    """

    # check types and values:
    assert is_pos_int(args.num_workers), "incorrect number of workers"
    assert is_pos_int(args.visdom_port), "incorrect visdom port"
    assert args.device == "cpu" or args.device == "gpu", "unknown device"

    # check that CUDA is available:
    if args.device == "gpu":
        assert torch.cuda.is_available(), "CUDA required to train on GPUs"

    # create checkpoint folder if it does not exist:
    if args.checkpoint_folder != "" and not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
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

    # set the right backend for torchvision:
    torchvision.set_image_backend("accimage")

    # return input arguments:
    return args


def get_parser():
    """
    Return a standard command-line parser.
    """
    parser = argparse.ArgumentParser(description="Train convolutional network")
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
