#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict

from . import ClassyParamScheduler, UpdateInterval, register_param_scheduler


@register_param_scheduler("cosine")
class CosineParamScheduler(ClassyParamScheduler):
    """
    Cosine decay or cosine warmup schedules based on start and end values.
    The schedule is updated after every train step by default based on the
    fraction of samples seen. The schedule was proposed in 'SGDR: Stochastic
    Gradient Descent with Warm Restarts' (https://arxiv.org/abs/1608.03983).
    Note that this class only implements the cosine annealing part of SGDR,
    and not the restarts.

    Example:

        .. code-block:: python

          start_value: 0.1
          end_value: 0.0001
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
    def from_config(cls, config: Dict[str, Any]) -> "CosineParamScheduler":
        """Instantiates a CosineParamScheduler from a configuration.

        Args:
            config: A configuration for a CosineParamScheduler.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A CosineParamScheduler instance.
        """
        assert (
            "start_value" in config and "end_value" in config
        ), "Cosine scheduler requires a start_value and a end_value"

        return cls(
            start_value=config["start_value"],
            end_value=config["end_value"],
            update_interval=UpdateInterval.from_config(config, UpdateInterval.STEP),
        )

    def __call__(self, where: float):
        return self._end_value + 0.5 * (self._start_value - self._end_value) * (
            1 + math.cos(math.pi * where)
        )
