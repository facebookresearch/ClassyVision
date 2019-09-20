#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class UpdateInterval(Enum):
    EPOCH = 1
    STEP = 2


class ClassyParamScheduler(object):
    # To be used for comparisons with where
    WHERE_EPSILON = 1e-6

    def __init__(self, config):
        self.update_interval = UpdateInterval.EPOCH

    def __call__(self, where: float):
        """
        Get the param for a given point at training. where is a float between
        [0;1) that specifies how far along we are;
        """
        raise NotImplementedError("Param schedulers must override __call__")
