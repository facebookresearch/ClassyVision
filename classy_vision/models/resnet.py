#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of ResNet (https://arxiv.org/pdf/1512.03385.pdf) as a special
case of ResNeXt (https://arxiv.org/pdf/1611.05431.pdf)
"""

from . import register_model
from .resnext import ResNeXt


# global setting for in-place ReLU:
INPLACE = True


@register_model("resnet")
class ResNet(ResNeXt):
    """
    ResNet is a special case of :class:`ResNeXt`.
    """

    def __init__(self, **kwargs):
        """
        See :func:`ResNeXt.__init__`
        """
        assert (
            kwargs["base_width_and_cardinality"] is None
        ), "base_width_and_cardinality should be None for ResNet"
        super().__init__(**kwargs)
