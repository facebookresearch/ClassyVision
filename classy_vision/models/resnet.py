#!/usr/bin/env python3

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
        ResNet is a special case of ResNeXt.
    """

    def __init__(self, config):
        config["base_width_and_cardinality"] = None
        super().__init__(config)
