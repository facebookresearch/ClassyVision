#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Dict


class UpdateInterval(Enum):
    """
    Enum for specifying update frequency for scheduler.

    Attributes:
        EPOCH (str): Update param before each epoch
        STEP (str): Update param before each optimizer step
    """

    EPOCH = "epoch"
    STEP = "step"


class ClassyParamScheduler(object):
    """
    Base class for Classy parameter schedulers.

    Attributes:
        update_interval: Specifies how often to update each parameter
            (before each epoch or each batch)
    """

    # To be used for comparisons with where
    WHERE_EPSILON = 1e-6

    def __init__(self, update_interval: UpdateInterval = UpdateInterval.EPOCH):
        """
        Constructor for ClassyParamScheduler

        Args:
            update_interval: Specifies the frequency of the param updates
        """
        self.update_interval = update_interval

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyParamScheduler":
        """Instantiates a ClassyParamScheduler from a configuration.

        Args:
            config: A configuration for the ClassyParamScheduler.

        Returns:
            A ClassyParamScheduler instance.
        """
        raise NotImplementedError

    def __call__(self, where: float):
        """
        Get the param for a given point at training.

        For Classy Vision we update params (such as learning rate) based on
        the percent progress of training completed. This allows a
        scheduler to be agnostic to the exact specifications of a
        particular run (e.g. 120 epochs vs 90 epochs).

        Args:
            where: A float in [0;1) that represents how far training has progressed

        """
        raise NotImplementedError("Param schedulers must override __call__")
