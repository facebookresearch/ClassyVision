#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class ClassyMeter(object):
    """
    Base class for meters to measure various metrics during
    training and testing phases.
    """

    def __init__(self, **kwargs):
        """Init function with kwargs passed for meter configuration.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    @property
    def name(self):
        """Name of the meter.
        """
        raise NotImplementedError

    @property
    def value(self):
        """Value of meter based on local state, can be any python object.

        Note: If there are multiple training processes then this
        represents the local state of the meter. If sync meter is
        implemented, then value will return the global state since the
        last sync PLUS any local unsynced updates that have occurred
        in the local process.
        """
        raise NotImplementedError

    def sync_state(self):
        """Syncs state with all other meters in distributed training.

        WARNING: Calls to sync_state could involve communications via
        torch.distributed which can result in a loss of performance or
        deadlocks if not coordinated among threads
        """
        # If not provided by child class this does nothing by default
        # and meter only provides the local process stats. If
        # implemented then the meter provides the global stats at last
        # sync + any local updates since the last sync
        pass

    def reset(self):
        """Resets any internal meter state.
        Gets called at the end of each phase.
        """
        raise NotImplementedError

    def update(self, model_output, target, **kwargs):
        """Updates any internal state of meter.
        Gets called after each batch processing of each phase.

        Args:
            model_output (torch.Tensor): Output of classy_model.
            target       (torch.Tensor): Target provided by dataloader.
        """
        raise NotImplementedError

    def validate(self, model_output_shape, target_shape):
        """Validates if the meter can be calculated on the given model_output_shape
        and target_shape.
        """
        raise NotImplementedError

    def get_classy_state(self):
        """Gets the state of the ClassyMeter.
        """
        raise NotImplementedError

    def set_classy_state(self, state):
        """Sets the state of the ClassyMeter.

        Args:
            state: State of the Meter to restore.
        """
        raise NotImplementedError
