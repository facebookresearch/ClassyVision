#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn


class ClassyBlock(nn.Module):
    """
    This is a thin wrapper for head execution, which records the output of
    wrapped module for executing the heads forked from this module.

    `block_outs` is received as the argument and gets returned with the current block name
    appended. Note, that we can't always rely on just modifying argument in place as it
    doesn't work on TorchScript/Python boundary.
    """

    def __init__(self, name, module):
        super().__init__()
        self.name = name
        self._module = module

    def forward(self, input, block_outs: Dict[str, torch.Tensor]):
        output = self._module(input)
        assert self.name not in block_outs
        block_outs[self.name] = output
        return output, block_outs


class BlockSequential(nn.Sequential):
    """
    Like nn.Sequential but for ClassyBlocks - it passes through additional argument `block_outs`
    """

    def forward(self, input, block_outs: Dict[str, torch.Tensor]):
        for module in self:
            input, block_outs = module(input, block_outs)
        return input, block_outs
