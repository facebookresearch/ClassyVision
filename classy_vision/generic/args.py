#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from classy_vision.generic.opts import check_generic_args, parse_train_arguments
from classy_vision.generic.util import load_json


try:
    import hydra

    hydra_available = True
except ImportError:
    hydra_available = False

args = None
config = None


if hydra_available:

    @hydra.main(config_path="../hydra/args.yaml")
    def _parse_hydra_args(cfg):
        # This need to be a separate function which sets globals because hydra doesn't
        # support returning from its main function
        global args, config
        args = cfg
        check_generic_args(args)
        config = args.config.to_container()


def parse_args():
    """Parse arguments.

    Parses the args from argparse. If hydra is installed, uses hydra based args
    (experimental).
    """
    if hydra_available:
        global args, config
        _parse_hydra_args()
        return args, config
    else:
        args = parse_train_arguments()
        config = load_json(args.config_file)
        return args, config
