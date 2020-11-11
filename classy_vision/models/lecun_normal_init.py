# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn as nn


def lecun_normal_init(tensor, fan_in):
    nn.init.trunc_normal_(tensor, std=math.sqrt(1 / fan_in))
