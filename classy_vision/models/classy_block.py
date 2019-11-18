#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class ClassyBlock(nn.Module):
    """
    This is a thin wrapper for head execution, which records the output of
    wrapped module for executing the heads forked from this module.
    """

    def __init__(self, name, module):
        super().__init__()
        self.name = name
        self.output = torch.zeros(0)
        self._module = module
        self._should_cache_output = False

    def set_cache_output(self, should_cache_output: bool = True):
        """
        Whether to cache the output of wrapped module for head execution.
        """
        self._should_cache_output = should_cache_output

    def forward(self, input):
        output = self._module(input)
        if self._should_cache_output:
            self.output = output
        return output
