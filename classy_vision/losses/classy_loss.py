#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.nn as nn


class ClassyLoss(nn.Module):
    """
    Base class to calculate the loss during training.

    This implementation of :class:`torch.nn.Module` allows building
    the loss object from a configuration file.
    """

    def __init__(self):
        """
        Constructor for ClassyLoss.
        """
        super(ClassyLoss, self).__init__()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyLoss":
        """Instantiates a ClassyLoss from a configuration.

        Args:
            config: A configuration for a ClassyLoss.

        Returns:
            A ClassyLoss instance.
        """
        raise NotImplementedError()

    def forward(self, output, target):
        """
        Compute the loss for the provided sample.

        Refer to :class:`torch.nn.Module` for more details.
        """
        raise NotImplementedError
