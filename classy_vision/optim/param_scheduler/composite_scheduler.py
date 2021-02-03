#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, auto
from typing import Any, Dict, Sequence, Union

from fvcore.common import param_scheduler

from . import (
    UpdateInterval,
    build_param_scheduler,
    register_param_scheduler,
)


class IntervalScaling(Enum):
    RESCALED = auto()
    FIXED = auto()


@register_param_scheduler("composite")
class CompositeParamScheduler(param_scheduler.CompositeParamScheduler):
    __doc__ = param_scheduler.CompositeParamScheduler.__doc__

    def __init__(
        self,
        schedulers: Sequence[param_scheduler.ParamScheduler],
        lengths: Sequence[float],
        interval_scaling: Sequence[Union[IntervalScaling, str]],
        update_interval: UpdateInterval = UpdateInterval.STEP,
    ):
        scaling_name = {
            IntervalScaling.RESCALED: "rescaled",
            IntervalScaling.FIXED: "fixed",
        }
        interval_scaling = [
            scaling_name[s] if isinstance(s, IntervalScaling) else s
            for s in interval_scaling
        ]
        super().__init__(schedulers, lengths, interval_scaling)
        self.update_interval = update_interval

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompositeParamScheduler":
        """Instantiates a CompositeParamScheduler from a configuration.

        Args:
            config: A configuration for a CompositeParamScheduler.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A CompositeParamScheduler instance.
        """
        assert (
            "schedulers" in config and "lengths" in config
        ), "Composite scheduler needs both a list of schedulers and lengths"
        interval_scaling = []
        if "interval_scaling" in config:
            assert len(config["schedulers"]) == len(
                config["interval_scaling"]
            ), "Schedulers and interval scaling must be the same length"
            for interval_scale in config["interval_scaling"]:
                assert interval_scale in {
                    "fixed",
                    "rescaled",
                }, "Choices for interval scaling are 'fixed' or 'rescaled'"
                interval_scaling.append(IntervalScaling[interval_scale.upper()])
        else:
            interval_scaling = [IntervalScaling.RESCALED] * len(config["schedulers"])
        if "num_epochs" in config:  # Propagate value to intermediate schedulers
            config["schedulers"] = [
                dict(schedule, **{"num_epochs": config["num_epochs"]})
                for schedule in config["schedulers"]
            ]
        return cls(
            schedulers=[
                build_param_scheduler(scheduler) for scheduler in config["schedulers"]
            ],
            lengths=config["lengths"],
            update_interval=UpdateInterval.from_config(config, UpdateInterval.STEP),
            interval_scaling=interval_scaling,
        )
