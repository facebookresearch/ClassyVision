#!/usr/bin/env python3

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict

from classy_vision.state.classy_state import ClassyState


class ClassyHookFunctions(Enum):
    """
    Enumeration of all the hook functions in the ClassyHook class.
    """

    on_rendezvous = auto()
    on_start = auto()
    on_phase_start = auto()
    on_sample = auto()
    on_forward = auto()
    on_loss = auto()
    on_backward = auto()
    on_update = auto()
    on_phase_end = auto()
    on_end = auto()


class ClassyHook(ABC):
    """
    Abstract class for hooks to plug in to the classy workflow at various points
    to add various functionalities such as logging and reporting.
    """

    def _noop(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Derived classes can set their hook functions to this if they want those
        hook functions to not do anything.
        """
        pass

    @abstractmethod
    def on_rendezvous(
        self, state: ClassyState, local_variables: Dict[str, Any]
    ) -> None:
        """
        Called when the trainers rendezvous.
        """
        pass

    @abstractmethod
    def on_start(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Called at the start of training.
        """
        pass

    @abstractmethod
    def on_phase_start(
        self, state: ClassyState, local_variables: Dict[str, Any]
    ) -> None:
        """
        Called at the start of each phase.
        """
        pass

    @abstractmethod
    def on_sample(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Called each time trainer obtained a sample.
        """
        pass

    @abstractmethod
    def on_forward(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Called each time forward pass is triggered.
        """
        pass

    @abstractmethod
    def on_loss(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Called each time a loss has been computed.
        """
        pass

    @abstractmethod
    def on_backward(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Called each time a backward step is performed on the loss.
        """
        pass

    @abstractmethod
    def on_update(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Called each time parameters have been updated.
        """
        pass

    @abstractmethod
    def on_phase_end(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Called at the end of each phase.
        """
        pass

    @abstractmethod
    def on_end(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Called at the end of training.
        """
        pass
