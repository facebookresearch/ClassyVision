#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torchvision.models as models
from classy_vision.models import ClassyModel, register_model


@register_model("my_model")
class MyModel(ClassyModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((20, 20)),
            nn.Flatten(1),
            nn.Linear(3 * 20 * 20, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls()
