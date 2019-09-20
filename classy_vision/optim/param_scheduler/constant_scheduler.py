#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import ClassyParamScheduler, register_param_scheduler


@register_param_scheduler("constant")
class ConstantParamScheduler(ClassyParamScheduler):
    """
    Returns a constant value for a optimizer param.
    """

    def __init__(self, config):
        super().__init__(config)
        assert "value" in config
        self._value = config["value"]

    def __call__(self, where: float):
        if where >= 1.0:
            raise RuntimeError(f"Invalid where parameter for scheduler: {where}")

        return self._value
