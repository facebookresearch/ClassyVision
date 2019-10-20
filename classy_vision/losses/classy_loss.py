#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class ClassyLoss(nn.Module):
    def __init__(self):
        """
        ClassyLoss constructor. This stores the loss configuration for
        future access and constructs the basic loss object.
        """
        super(ClassyLoss, self).__init__()

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError()

    def forward(self, output, target):
        """
        Compute the loss for the provided sample.
        """
        raise NotImplementedError
