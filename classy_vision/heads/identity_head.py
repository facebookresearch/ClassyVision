#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

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
    def from_config(cls, config: Dict[str, Any]) -> "IdentityHead":
        """Instantiates a IdentityHead from a configuration.

        Args:
            config: A configuration for a IdentityHead.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A IdentityHead instance.
        """
        return cls(config["unique_id"])
