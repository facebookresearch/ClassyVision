#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from classy_vision.generic.util import is_pos_int
from classy_vision.heads import ClassyHead, register_head


@register_head("fully_connected")
class FullyConnectedHead(ClassyHead):
    def __init__(self, unique_id, num_classes, in_plane):
        super().__init__(unique_id, num_classes)
        assert num_classes is None or is_pos_int(num_classes)
        assert is_pos_int(in_plane)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None if num_classes is None else nn.Linear(in_plane, num_classes)

    @classmethod
    def from_config(cls, config):
        num_classes = config.get("num_classes", None)
        in_plane = config["in_plane"]
        return cls(config["unique_id"], num_classes, in_plane)

    def forward(self, x):
        # perform average pooling:
        out = self.avgpool(x)

        # final classifier:
        out = out.reshape(out.size(0), -1)
        if self.fc is not None:
            out = self.fc(out)
        return out
