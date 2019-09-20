#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn.modules.loss import _WeightedLoss


class ClassyCriterion(_WeightedLoss):
    def __init__(self, config):
        """
        Classy Criterion constructor. This stores the criterion config for
        future access and constructs the basic criterion object.
        """
        super(ClassyCriterion, self).__init__()
        self._config = config

    def forward(self, output, target):
        """
        Compute the loss for the provided sample.
        """
        raise NotImplementedError
