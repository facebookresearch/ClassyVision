#!/usr/bin/env python3

from . import ClassyParamScheduler, register_param_scheduler


@register_param_scheduler("polynomial")
class PolynomialDecayParamScheduler(ClassyParamScheduler):
    """
    Decays the param value after every epoch according to a
    polynomial function with a fixed power.

    Example:
      base_lr: 0.1
      power: 0.9

    Then the param value will be 0.1 for epoch 0, 0.099 for epoch 1, and
    so on.
    """

    def __init__(self, config):
        super().__init__(config)
        assert (
            "base_lr" in config and "power" in config
        ), "Polynomial decay scheduler requires a base lr and a power of decay"

        self._base_lr = config["base_lr"]
        self._power = config["power"]

    def __call__(self, where: float):
        return self._base_lr * (1 - where) ** self._power
