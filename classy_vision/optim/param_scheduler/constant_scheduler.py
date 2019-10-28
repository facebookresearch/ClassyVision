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

    def __init__(self, value: float, update_interval: str):
        super().__init__(update_interval)
        self._value = value

    @classmethod
    def from_config(cls, config):
        assert "value" in config
        update_interval = "epoch"
        if "update_interval" in config:
            update_interval = config["update_interval"]
        return cls(value=config["value"], update_interval=update_interval)

    def __call__(self, where: float):
        if where >= 1.0:
            raise RuntimeError(f"Invalid where parameter for scheduler: {where}")

        return self._value
