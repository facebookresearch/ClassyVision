#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Any, Dict


class ClassyTask(ABC):
    def __init__(self):
        self.hooks = []

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        raise NotImplementedError()

    @abstractmethod
    def init_distributed_data_parallel_model(self):
        pass

    @property
    @abstractmethod
    def where(self):
        pass

    @abstractmethod
    def advance_phase(self):
        pass

    @abstractmethod
    def done_training(self):
        pass

    @abstractmethod
    def get_classy_state(self, deep_copy=False):
        """
        Returns a dictionary containing the state stored inside the object.

        If deep_copy is True (default False), creates a deep copy. Otherwise,
        the returned dict's attributes will be tied to the object's.
        """
        pass

    @abstractmethod
    def set_classy_state(self, state):
        pass

    @abstractmethod
    def prepare(self, num_workers=0, pin_memory=False, use_gpu=False):
        """
        Prepares the task for training.
        """
        pass

    @abstractmethod
    def train_step(self, use_gpu, local_variables=None):
        pass

    def run_hooks(self, local_variables: Dict[str, Any], hook_function: str) -> None:
        """
        Helper function that runs hook_function for all the classy hooks.
        """
        for hook in self.hooks:
            getattr(hook, hook_function)(self, local_variables)
