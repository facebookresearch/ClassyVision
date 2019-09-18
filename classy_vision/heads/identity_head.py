#!/usr/bin/env python3

from classy_vision.heads import register_head
from classy_vision.heads.classy_vision_head import ClassyVisionHead


@register_head("identity")
class IdentityHead(ClassyVisionHead):
    def __init__(self, head_config):
        super().__init__(head_config)

    def forward(self, x):
        return x
