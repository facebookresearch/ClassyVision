#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.models as models
from classy_vision.models import ClassyModel, register_model


@register_model("my_model")
class MyModel(ClassyModel):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18()

    def forward(self, x):
        return self.resnet(x)

    @classmethod
    def from_config(cls, config):
        return cls()
