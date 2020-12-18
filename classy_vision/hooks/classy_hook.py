#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict

from classy_vision.generic.util import log_class_usage


class ClassyHookState:
    """Class to store state within instances of ClassyHook.

    Any serializable data can be stored in the instance's attributes.
    """

    def get_classy_state(self) -> Dict[str, Any]:
        return self.__dict__

    def set_classy_state(self, state_dict: Dict[str, Any]):
        # We take a conservative approach and only update the dictionary instead of
        # replacing it. This allows hooks to continue functioning in case the state
        # is loaded from older implementations.
        self.__dict__.update(state_dict)


class ClassyHook(ABC):
    """Base class for hooks.

    Hooks allow to inject behavior at different places of the training loop, which
    are listed below in the chronological order.

        on_start -> on_phase_start ->
            on_step -> on_phase_end -> on_end

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
        log_class_usage("Hooks", self.__class__)
        self.state = ClassyHookState()

    @classmethod
    def from_config(cls, config) -> "ClassyHook":
        return cls(**config)

    def _noop(self, *args, **kwargs) -> None:
        """Derived classes can set their hook functions to this.

        This is useful if they want those hook functions to not do anything.

        """
        pass

    @classmethod
    def name(cls) -> str:
        """Returns the name of the class."""
        return cls.__name__

    @abstractmethod
    def on_start(self, task) -> None:
        """Called at the start of training."""
        pass

    @abstractmethod
    def on_phase_start(self, task) -> None:
        """Called at the start of each phase."""
        pass

    @abstractmethod
    def on_step(self, task) -> None:
        """Called each time after parameters have been updated by the optimizer."""
        pass

    @abstractmethod
    def on_phase_end(self, task) -> None:
        """Called at the end of each phase (epoch)."""
        pass

    @abstractmethod
    def on_end(self, task) -> None:
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
