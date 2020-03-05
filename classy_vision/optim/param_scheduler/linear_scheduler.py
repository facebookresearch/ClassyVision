#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from . import ClassyParamScheduler, UpdateInterval, register_param_scheduler


@register_param_scheduler("linear")
class LinearParamScheduler(ClassyParamScheduler):
    """
    Linearly interpolates parameter between ``start_value`` and ``end_value``.
    Can be used for either warmup or decay based on start and end values.
    The schedule is updated after every train step by default.

    Example:

        .. code-block:: python

            start_value: 0.0001
            end_value: 0.01
    Corresponds to a linear increasing schedule with values in [0.0001, 0.01)
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        update_interval: UpdateInterval = UpdateInterval.STEP,
    ):
        super().__init__(update_interval=update_interval)
        self._start_value = start_value
        self._end_value = end_value

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
            "start_value" in config and "end_value" in config
        ), "Linear scheduler requires a start and a end"

        return cls(
            start_value=config["start_value"],
            end_value=config["end_value"],
            update_interval=UpdateInterval.from_config(config, UpdateInterval.STEP),
        )

    def __call__(self, where: float):
        # interpolate between start and end values
        return self._end_value * where + self._start_value * (1 - where)
