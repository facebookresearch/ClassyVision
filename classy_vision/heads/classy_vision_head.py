#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch.nn as nn


class ClassyVisionHead(nn.Module):
    def __init__(
        self, unique_id: Optional[str] = None, num_classes: Optional[int] = None
    ):
        """
        Classy Head constructor.
        This is the place to build and initialize the layers.
        """
        super().__init__()
        self.unique_id = unique_id or self.__class__.__name__
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
