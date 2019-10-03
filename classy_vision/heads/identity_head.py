#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from classy_vision.heads import ClassyVisionHead, register_head


@register_head("identity")
class IdentityHead(ClassyVisionHead):
    def forward(self, x):
        return x

    @classmethod
    def from_config(cls, config):
        return cls()
