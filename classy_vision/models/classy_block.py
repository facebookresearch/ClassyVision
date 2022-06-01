#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

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
        # `ClassyBlock` isn't thread safe since it saves state. To avoid
        # doing this, the recommended workflow is to set `Model.wrapper_cls = None`
        # before instantiation (see the docs for `ClassyModel`). We support this
        # environment variable for older use cases but using it is not recommended.
        self._is_output_stateless = os.environ.get("CLASSY_BLOCK_STATELESS") == "1"

    def wrapped_module(self):
        return self._module

    def forward(self, input):
        if hasattr(self, "_is_output_stateless"):
            if self._is_output_stateless:
                return self._module(input)

        output = self._module(input)
        self.output = output
        return output
