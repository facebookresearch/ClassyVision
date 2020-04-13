#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.nn as nn
from classy_vision.generic.util import is_pos_int
from classy_vision.heads import ClassyHead, register_head


@register_head("fully_connected")
class FullyConnectedHead(ClassyHead):
    """This head defines a 2d average pooling layer
    (:class:`torch.nn.AdaptiveAvgPool2d`) followed by a fully connected
    layer (:class:`torch.nn.Linear`).
    """

    def __init__(
        self,
        unique_id: str,
        num_classes: int,
        in_plane: int,
        zero_init_bias: bool = False,
    ):
        """Constructor for FullyConnectedHead

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.

            num_classes: Number of classes for the head. If None, then the fully
                connected layer is not applied.

            in_plane: Input size for the fully connected layer.
        """
        super().__init__(unique_id, num_classes)
        assert num_classes is None or is_pos_int(num_classes)
        assert is_pos_int(in_plane)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None if num_classes is None else nn.Linear(in_plane, num_classes)

        if zero_init_bias:
            self.fc.bias.data.zero_()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FullyConnectedHead":
        """Instantiates a FullyConnectedHead from a configuration.

        Args:
            config: A configuration for a FullyConnectedHead.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A FullyConnectedHead instance.
        """
        num_classes = config.get("num_classes", None)
        in_plane = config["in_plane"]
        return cls(
            config["unique_id"],
            num_classes,
            in_plane,
            zero_init_bias=config.get("zero_init_bias", False),
        )

    def forward(self, x):
        # perform average pooling:
        out = self.avgpool(x)

        # final classifier:
        out = out.flatten(start_dim=1)
        if self.fc is not None:
            out = self.fc(out)
        return out
