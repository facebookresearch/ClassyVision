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

import torch.nn as nn
from classy_vision.heads import ClassyHead, register_head

from ..models.lecun_normal_init import lecun_normal_init


@register_head("vision_transformer_head")
class VisionTransformerHead(ClassyHead):
    def __init__(
        self,
        in_plane,
        num_classes,
        hidden_dim=None,
    ):
        super().__init__()
        if hidden_dim is None:
            layers = [("head", nn.Linear(in_plane, num_classes))]
        else:
            layers = [
                ("pre_logits", nn.Linear(in_plane, hidden_dim)),
                ("act", nn.Tanh()),
                ("head", nn.Linear(hidden_dim, num_classes)),
            ]
        self.layers = nn.Sequential(OrderedDict(layers))
        self.init_weights()

    def init_weights(self):
        if hasattr(self.layers, "pre_logits"):
            lecun_normal_init(
                self.layers.pre_logits.weight, fan_in=self.layers.pre_logits.in_features
            )
            nn.init.zeros_(self.layers.pre_logits.bias)
        nn.init.zeros_(self.layers.head.weight)
        nn.init.zeros_(self.layers.head.bias)

    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        config.pop("unique_id")
        return cls(**config)

    def forward(self, x):
        return self.layers(x)
