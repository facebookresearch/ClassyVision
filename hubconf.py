#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from classy_vision.hub import ClassyHubInterface


dependencies = ["torch", "torchvision"]

# export the wsl models (https://github.com/facebookresearch/WSL-Images)
resnext_wsl_models = [
    "resnext101_32x8d_wsl",
    "resnext101_32x16d_wsl",
    "resnext101_32x32d_wsl",
    "resnext101_32x48d_wsl",
]


def _create_interface_from_torchhub(github, *args, **kwargs):
    model = torch.hub.load(github, *args, **kwargs)
    return ClassyHubInterface.from_model(model)


for model in resnext_wsl_models:
    globals()[model] = functools.partial(
        _create_interface_from_torchhub, "facebookresearch/WSL-Images", model
    )
