#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch
from classy_vision.generic.util import log_class_usage


class ClassyMeter:
    """
    Base class to measure various metrics during training and testing phases.

    This can include meters like  Accuracy, Precision and Recall, etc.
    """

    def __init__(self):
        log_class_usage("Meter", self.__class__)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyMeter":
        """Instantiates a ClassyMeter using a configuration.

        Args:
            config: A configuration for a ClassyMeter.

        Returns:
            A ClassyMeter instance.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """The name of the meter."""
        raise NotImplementedError

    @property
    def value(self) -> Any:
        """
        Value of meter based on local state, can be any python object.

        Note:
            If there are multiple training processes then this
            represents the local state of the meter. If :func:`sync_state` is
            implemented, then value will return the global state since the
            last sync PLUS any local unsynced updates that have occurred
            in the local process.
        """
        raise NotImplementedError

    def sync_state(self) -> None:
        """
        Syncs state with all other meters in distributed training.

        If not provided by child class this does nothing by default
        and meter only provides the local process stats. If
        implemented then the meter provides the global stats at last
        sync + any local updates since the last sync.

        Warning:
            Calls to sync_state could involve communications via
            :mod:`torch.distributed` which can result in a loss of performance or
            deadlocks if not coordinated among threads.
        """
        pass

    def reset(self):
        """
        Resets any internal meter state.

        Should normally be called at the end of a phase.
        """
        raise NotImplementedError

    def update(
        self, model_output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> None:
        """
        Updates any internal state of meter.

        Should be called after each batch processing of each phase.

        Args:
            model_output: Output of a :class:`ClassyModel`.
            target: Target provided by a dataloader from :class:`ClassyDataset`.
        """
        raise NotImplementedError

    def validate(self, model_output_shape: Tuple, target_shape: Tuple) -> None:
        """
        Validate the meter.

        Checks if the meter can be calculated on the given ``model_output_shape``
        and ``target_shape``.
        """
        raise NotImplementedError

    def get_classy_state(self) -> Dict[str, Any]:
        """Get the state of the ClassyMeter.

        The returned state is used for checkpointing.

        Returns:
            A state dictionary containing the state of the meter.
        """
        raise NotImplementedError

    def set_classy_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the ClassyMeter.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the meter from a checkpoint.
        """
        raise NotImplementedError

    def __repr__(self):
        """Returns a string representation of the meter, used for logging.

        The default implementation assumes value is a dict. value is not
        required to be a dict, and in that case you should override this
        method."""

        if not isinstance(self.value, dict):
            return super().__repr__()

        values = ",".join([f"{key}={value:.6f}" for key, value in self.value.items()])
        return f"{self.name}_meter({values})"
