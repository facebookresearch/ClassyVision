#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Vision Transformer head implementation from https://arxiv.org/abs/2010.11929.

References:
https://github.com/google-research/vision_transformer
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import copy
from collections import OrderedDict
from typing import Optional

import torch.nn as nn
from classy_vision.heads import ClassyHead, register_head

from ..models.lecun_normal_init import lecun_normal_init


NORMALIZE_L2 = "l2"


@register_head("vision_transformer_head")
class VisionTransformerHead(ClassyHead):
    def __init__(
        self,
        unique_id: str,
        in_plane: int,
        num_classes: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        normalize_inputs: Optional[str] = None,
    ):
        """
        Args:
            unique_id: A unique identifier for the head
            in_plane: Input size for the fully connected layer
            num_classes: Number of output classes for the head
            hidden_dim: If not None, a hidden layer with the specific dimension is added
            normalize_inputs: If specified, normalize the inputs using the specified
                method. Supports "l2" normalization.
        """
        super().__init__(unique_id, num_classes)

        if normalize_inputs is not None and normalize_inputs != NORMALIZE_L2:
            raise ValueError(
                f"Unsupported value for normalize_inputs: {normalize_inputs}"
            )

        if num_classes is None:
            layers = []
        elif hidden_dim is None:
            layers = [("head", nn.Linear(in_plane, num_classes))]
        else:
            layers = [
                ("pre_logits", nn.Linear(in_plane, hidden_dim)),
                ("act", nn.Tanh()),
                ("head", nn.Linear(hidden_dim, num_classes)),
            ]
        self.layers = nn.Sequential(OrderedDict(layers))
        self.normalize_inputs = normalize_inputs
        self.init_weights()

    def init_weights(self):
        if hasattr(self.layers, "pre_logits"):
            lecun_normal_init(
                self.layers.pre_logits.weight, fan_in=self.layers.pre_logits.in_features
            )
            nn.init.zeros_(self.layers.pre_logits.bias)
        if hasattr(self.layers, "head"):
            nn.init.zeros_(self.layers.head.weight)
            nn.init.zeros_(self.layers.head.bias)

    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        return cls(**config)

    def forward(self, x):
        if self.normalize_inputs is not None:
            if self.normalize_inputs == NORMALIZE_L2:
                x = nn.functional.normalize(x, p=2.0, dim=1)
        return self.layers(x)
