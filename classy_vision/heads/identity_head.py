#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from classy_vision.heads import ClassyHead, register_head


@register_head("identity")
class IdentityHead(ClassyHead):
    """This head returns the input without changing it. This can
    be attached to a model, if the output of the model is the
    desired result.
    """

    def forward(self, x):
        return x

    @classmethod
    def from_config(cls, config):
        return cls(config["unique_id"])
