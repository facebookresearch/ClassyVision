#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from classy_vision.heads import ClassyVisionHead, register_head


@register_head("identity")
class IdentityHead(ClassyVisionHead):
    def __init__(self, head_config):
        super().__init__(head_config)

    def forward(self, x):
        return x
