#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from . import ClassyParamScheduler, register_param_scheduler


@register_param_scheduler("polynomial")
class PolynomialDecayParamScheduler(ClassyParamScheduler):
    """
    Decays the param value after every epoch according to a
    polynomial function with a fixed power.

    Example:

        .. code-block:: python

          base_lr: 0.1
          power: 0.9

    Then the param value will be 0.1 for epoch 0, 0.099 for epoch 1, and
    so on.
    """

    def __init__(self, base_lr, power):
        super().__init__()

        self._base_lr = base_lr
        self._power = power

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PolynomialDecayParamScheduler":
        """Instantiates a PolynomialDecayParamScheduler from a configuration.

        Args:
            config: A configuration for a PolynomialDecayParamScheduler.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A PolynomialDecayParamScheduler instance.
        """
        assert (
            "base_lr" in config and "power" in config
        ), "Polynomial decay scheduler requires a base lr and a power of decay"
        return cls(base_lr=config["base_lr"], power=config["power"])

    def __call__(self, where: float):
        return self._base_lr * (1 - where) ** self._power
