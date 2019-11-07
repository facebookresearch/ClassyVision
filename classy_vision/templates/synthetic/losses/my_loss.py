#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from classy_vision.losses import ClassyLoss, register_loss


@register_loss("my_loss")
class MyLoss(ClassyLoss):
    def forward(self, input, target):
        return F.cross_entropy(input, target)

    @classmethod
    def from_config(cls, config):
        # We don't need anything from the config
        return cls()
