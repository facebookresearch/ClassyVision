#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch.nn as nn


class ClassyHead(nn.Module):
    """
    Base class for heads that can be attached to :class:`ClassyModel`.

    A head is a regular :class:`torch.nn.Module` that can be attached to a
    pretrained model. This enables a form of transfer learning: utilizing a
    model trained for one dataset to extract features that can be used for
    other problems. A head must be attached to a :class:`models.ClassyBlock`
    within a :class:`models.ClassyModel`.
    """

    def __init__(
        self, unique_id: Optional[str] = None, num_classes: Optional[int] = None
    ):
        """
        Constructs a ClassyHead.

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.

            num_classes: Number of classes for the head.
        """
        super().__init__()
        self.unique_id = unique_id or self.__class__.__name__
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyHead":
        """Instantiates a ClassyHead from a configuration.

        Args:
            config: A configuration for the ClassyHead.

        Returns:
            A ClassyHead instance.
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Performs inference on the head.

        This is a regular PyTorch method, refer to :class:`torch.nn.Module` for
        more details
        """
        raise NotImplementedError
