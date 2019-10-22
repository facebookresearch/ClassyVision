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
        """Value of meter, can be any python object.
        """
        raise NotImplementedError

    @property
    def meter_state_dict(self):
        """Dictionary of states for the meter.
        """
        raise NotImplementedError

    @meter_state_dict.setter
    def meter_state_dict(self, classy_state):
        """Sets the meter state from classy_state.

        Args:
            classy_state: State to restore from.
        """
        raise NotImplementedError

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
        return self.meter_state_dict

    def set_classy_state(self, state):
        """Sets the state of the ClassyMeter.

        Args:
            state: State of the Meter to restore.
        """
        # Validate name and existence of mandatory fields (common to all meters).
        for field in self.meter_state_dict.keys():
            assert (
                field in state
            ), "State does not contain mandatory {0} " "field".format(field)
        assert self.name == state["name"], "Incompatible meter name!"

        # Assign classy_state to meter_state_dict.
        # Each meter runs its own assertions and then sets state.
        self.meter_state_dict = state
