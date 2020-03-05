#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, NamedTuple, Optional, Union

from . import ClassyParamScheduler, UpdateInterval, register_param_scheduler


@register_param_scheduler("step")
class StepParamScheduler(ClassyParamScheduler):
    """
    Takes a fixed schedule for a param value.  If the length of the
    fixed schedule is less than the number of epochs, then the epochs
    are divided evenly among the param schedule.
    The schedule is updated after every train epoch by default.

    Example:

        .. code-block:: python

          values: [0.1, 0.01, 0.001, 0.0001]
          num_epochs = 120

    Then the param value will be 0.1 for epochs 0-29, 0.01 for
    epochs 30-59, 0.001 for epoch 60-89, 0.0001 for epochs 90-119.
    """

    def __init__(
        self,
        num_epochs: Union[int, float],
        values: List[float],
        update_interval: UpdateInterval = UpdateInterval.EPOCH,
    ):
        super().__init__(update_interval=update_interval)

        self._param_schedule = values

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StepParamScheduler":
        """Instantiates a StepParamScheduler from a configuration.

        Args:
            config: A configuration for a StepParamScheduler.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A StepParamScheduler instance.
        """
        assert (
            "values" in config
            and isinstance(config["values"], list)
            and len(config["values"]) > 0
        ), "Step scheduler requires a list of at least one param value"
        assert config["num_epochs"] > 0, "Num epochs must be greater than 0"

        return cls(
            num_epochs=config["num_epochs"],
            values=config["values"],
            update_interval=UpdateInterval.from_config(config, UpdateInterval.EPOCH),
        )

    def __call__(self, where: float):
        ind = int((where + self.WHERE_EPSILON) * len(self._param_schedule))
        return self._param_schedule[ind]
