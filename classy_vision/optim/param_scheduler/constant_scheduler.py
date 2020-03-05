#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from . import ClassyParamScheduler, UpdateInterval, register_param_scheduler


@register_param_scheduler("constant")
class ConstantParamScheduler(ClassyParamScheduler):
    """
    Returns a constant value for a optimizer param.
    """

    def __init__(self, value: float):
        super().__init__(update_interval=UpdateInterval.EPOCH)
        self._value = value

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConstantParamScheduler":
        """Instantiates a ConstantParamScheduler from a configuration.

        Args:
            config: A configuration for a ConstantParamScheduler.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A ConstantParamScheduler instance.
        """
        assert "value" in config
        return cls(value=config["value"])

    def __call__(self, where: float):
        if where >= 1.0:
            raise RuntimeError(f"Invalid where parameter for scheduler: {where}")

        return self._value
