#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from . import ClassyParamScheduler, register_param_scheduler


@register_param_scheduler("linear")
class LinearParamScheduler(ClassyParamScheduler):
    """
    Linearly interpolates parameter between ``start_lr`` and ``end_lr``.
    Can be used for either warmup or decay based on start and end values.

    Example:

        .. code-block:: python

            start_lr: 0.0001
            end_lr: 0.01
    Corresponds to a linear increasing schedule with values in [0.0001, 0.01)
    """

    def __init__(self, start_lr: float, end_lr: float):
        super().__init__()
        self._start_lr = start_lr
        self._end_lr = end_lr

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LinearParamScheduler":
        """Instantiates a LinearParamScheduler from a configuration.

        Args:
            config: A configuration for a LinearParamScheduler.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A LinearParamScheduler instance.
        """
        assert (
            "start_lr" in config and "end_lr" in config
        ), "Linear scheduler requires a start and a end"
        return cls(start_lr=config["start_lr"], end_lr=config["end_lr"])

    def __call__(self, where: float):
        # interpolate between start and end values
        return self._end_lr * where + self._start_lr * (1 - where)
