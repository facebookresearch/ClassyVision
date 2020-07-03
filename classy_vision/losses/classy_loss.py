#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.nn as nn


class ClassyLoss(nn.Module):
    """
    Base class to calculate the loss during training.

    This implementation of :class:`torch.nn.Module` allows building
    the loss object from a configuration file.
    """

    def __init__(self):
        """
        Constructor for ClassyLoss.
        """
        super(ClassyLoss, self).__init__()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyLoss":
        """Instantiates a ClassyLoss from a configuration.

        Args:
            config: A configuration for a ClassyLoss.

        Returns:
            A ClassyLoss instance.
        """
        raise NotImplementedError()

    def forward(self, output, target):
        """
        Compute the loss for the provided sample.

        Refer to :class:`torch.nn.Module` for more details.
        """
        raise NotImplementedError

    def get_classy_state(self) -> Dict[str, Any]:
        """Get the state of the ClassyLoss.

        The returned state is used for checkpointing. Note that most losses are
        stateless and do not need to save any state.

        Returns:
            A state dictionary containing the state of the loss.
        """
        return self.state_dict()

    def set_classy_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the ClassyLoss.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the loss from a checkpoint. Note
        that most losses are stateless and do not need to load any state.
        """
        return self.load_state_dict(state)

    def has_learned_parameters(self) -> bool:
        """Does this loss have learned parameters?"""
        return any(param.requires_grad for param in self.parameters(recurse=True))
