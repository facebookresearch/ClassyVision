#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict

from classy_vision import tasks


class ClassyHookFunctions(Enum):
    """
    Enumeration of all the hook functions in the ClassyHook class.
    """

    on_rendezvous = auto()
    on_start = auto()
    on_phase_start = auto()
    on_sample = auto()
    on_forward = auto()
    on_loss_and_meter = auto()
    on_backward = auto()
    on_update = auto()
    on_phase_end = auto()
    on_end = auto()


class ClassyHookState:
    """Class to store state within instances of ClassyHook.

    Any serializable data can be stored in the instance's attributes.
    """

    def get_classy_state(self) -> Dict[str, Any]:
        return self.__dict__

    def set_classy_state(self, state_dict: Dict[str, Any]):
        self.__dict__ = state_dict


class ClassyHook(ABC):
    """Base class for hooks.

    Hooks allow to inject behavior at different places of the training loop, which
    are listed below in the chronological order.

        on_start -> on_phase_start -> on_sample -> on_forward -> on_loss_and_meter ->
            on_backward -> on_update -> on_phase_end -> on_end

    Deriving classes should call ``super().__init__()`` and store any state in
    ``self.state``. Any state added to this property should be serializable.
    E.g. -

    .. code-block:: python

        class MyHook(ClassyHook):
            def __init__(self, a, b):
                super().__init__()
                self.state.a = [1,2,3]
                self.state.b = "my_hook"
                # the following line is not allowed
                # self.state.my_lambda = lambda x: x^2

    """

    def __init__(self):
        self.state = ClassyHookState()

    def _noop(self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]) -> None:
        """Derived classes can set their hook functions to this.

        This is useful if they want those hook functions to not do anything.

        """
        pass

    @classmethod
    def name(cls) -> str:
        """Returns the name of the class."""
        return cls.__name__

    @abstractmethod
    def on_rendezvous(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Called when the trainers rendezvous."""
        pass

    @abstractmethod
    def on_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Called at the start of training."""
        pass

    @abstractmethod
    def on_phase_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Called at the start of each phase."""
        pass

    @abstractmethod
    def on_sample(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Called each time trainer obtained a sample from the dataset."""
        pass

    @abstractmethod
    def on_forward(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Called each time forward pass is done in the model."""
        pass

    @abstractmethod
    def on_loss_and_meter(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Called each time after a loss has been computed and meters are updated."""
        pass

    @abstractmethod
    def on_backward(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Called each time a backward step is performed on the loss."""
        pass

    @abstractmethod
    def on_update(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Called each time after parameters have been updated by the optimizer."""
        pass

    @abstractmethod
    def on_phase_end(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Called at the end of each phase (epoch)."""
        pass

    @abstractmethod
    def on_end(self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]) -> None:
        """Called at the end of training."""
        pass

    def get_classy_state(self) -> Dict[str, Any]:
        """Get the state of the ClassyHook.

        The returned state is used for checkpointing.

        Returns:
            A state dictionary containing the state of the hook.\

        """
        return self.state.get_classy_state()

    def set_classy_state(self, state_dict: Dict[str, Any]) -> None:
        """Set the state of the ClassyHook.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the hook from a checkpoint.

        """
        self.state.set_classy_state(state_dict)
