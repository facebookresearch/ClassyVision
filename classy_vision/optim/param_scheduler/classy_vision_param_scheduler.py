#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Dict

from classy_vision.generic.util import log_class_usage
from fvcore.common import param_scheduler


class UpdateInterval(Enum):
    """
    Enum for specifying update frequency for scheduler.

    Attributes:
        EPOCH (str): Update param before each epoch
        STEP (str): Update param before each optimizer step
    """

    EPOCH = "epoch"
    STEP = "step"

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], default: "UpdateInterval" = None
    ) -> "UpdateInterval":
        """Fetches the update interval from a config

        Args:
            config: The config for the parameter scheduler
            default: The value to use if the config doesn't specify an update interval.
                If not set, STEP is used.
        """
        if default is None:
            default = cls.STEP
        if "update_interval" not in config:
            return default
        if config.get("update_interval").lower() not in ["step", "epoch"]:
            raise ValueError("Choices for update interval are 'step' or 'epoch'")
        return cls[config["update_interval"].upper()]


class ClassyParamScheduler(param_scheduler.ParamScheduler):
    """
    Base class for Classy parameter schedulers.

    Attributes:
        update_interval: Specifies how often to update each parameter
            (before each epoch or each batch)
    """

    def __init__(self, update_interval: UpdateInterval):
        """
        Constructor for ClassyParamScheduler

        Args:
            update_interval: Specifies the frequency of the param updates
        """
        self.update_interval = update_interval
        log_class_usage("ParamScheduler", self.__class__)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyParamScheduler":
        """Instantiates a ClassyParamScheduler from a configuration.

        Args:
            config: A configuration for the ClassyParamScheduler.

        Returns:
            A ClassyParamScheduler instance.
        """
        raise NotImplementedError
