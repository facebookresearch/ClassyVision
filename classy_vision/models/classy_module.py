#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Dict

import torch.nn as nn
from classy_vision.heads import ClassyVisionHead
from torch import Tensor


class ClassyModule(nn.Module):
    def __init__(self, name, module):
        super().__init__()
        self._name = name
        self._module = module
        self._heads = nn.ModuleDict()
        self._head_outputs = {}

    @property
    def name(self):
        return self._name

    @property
    def head_outputs(self):
        return copy.copy(self._head_outputs)

    def load_head_states(self, head_states):
        """
        load state dict for all the heads
        Args:
            head_states (dict): mapping between head id and state dict
        """
        assert (
            len(head_states) == 0 or len(self._heads) != 0
        ), "Expect the heads to be constructed before loading the states"

        for head_id, head_state in head_states.items():
            self._heads[head_id].load_state_dict(head_state)

    def set_heads(self, heads):
        """
        attach heads to current module.
        Args:
            heads (list): a list of ClassyVisionHead
        """
        if not all(isinstance(x, ClassyVisionHead) for x in heads):
            raise ValueError("Head must extend ClassyVisionHead")

        self._heads = nn.ModuleDict()
        for head in heads:
            self._heads[head.unique_id] = head

    def get_heads(self):
        return dict(self._heads)

    def forward(self, input: Tensor) -> Tensor:
        input = self._module(input)
        for head in self._heads.values():
            assert (
                not head.requires_dict_input()
            ), f"The head {head.unique_id} expects dict input"
            self._head_outputs[head.unique_id] = head(input)
        return input


class ClassyModuleWithDictInput(ClassyModule):
    def forward(self, input: Dict) -> Dict:
        assert isinstance(input, dict)
        input = copy.deepcopy(input)
        input["data"] = self._module(input["data"])
        for head in self._heads.values():
            if head.requires_dict_input():
                self._head_outputs[head.unique_id] = head(input)
            else:
                self._head_outputs[head.unique_id] = head(input["data"])
        return input
