#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ClassyTask(ABC):
    """
    An abstract base class for a training task.

    A ClassyTask encapsulates all the components and steps needed
    to train using a :func:`ClassyTrainer`.
    """

    def __init__(self) -> "ClassyTask":
        """
        Constructs a ClassyTask.
        """
        self.hooks = []

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyTask":
        raise NotImplementedError()

    @abstractmethod
    def init_distributed_data_parallel_model(self) -> None:
        """
        Initialize :class:`torch.nn.parallel.distributed.DistributedDataParallel`.

        Needed for distributed training. This is where a model should be wrapped by DDP.
        """
        pass

    @property
    @abstractmethod
    def where(self) -> float:
        """
        Tells how far along (where) we are during training.

        Returns:
            A float in [0, 1) which tells the training progress.
        """
        pass

    @abstractmethod
    def advance_phase(self) -> None:
        """
        Advance the task a phase.

        Called when one phase of reading from :class:`ClassyDataset` is over.
        """
        pass

    @abstractmethod
    def done_training(self) -> bool:
        """
        Tells if we are done training.

        Returns:
            A boolean telling if training is over.
        """
        pass

    @abstractmethod
    def get_classy_state(self, deep_copy: bool = False) -> Dict[str, Any]:
        """
        Get the state of the ClassyTask.

        Args:
            deep_copy: Creates a deep copy of the state if True (default is False).
                Otherwise, the returned dict's attributes will be tied to the object's.

        Returns:
            A state dictionary containing the state of the task.
        """
        pass

    @abstractmethod
    def set_classy_state(self, state):
        """
        Set the state of the ClassyTask.

        Args:
            state: State of the task to restore.
        """
        pass

    @abstractmethod
    def prepare(
        self, num_dataloader_workers=0, pin_memory=False, use_gpu=False
    ) -> None:
        """
        Prepares the task for training.

        Will be called by the :class:`ClassyTrainer` to prepare the task.

        Args:
            num_dataloader_workers: Number of workers to create for the dataloaders
            pin_memory: Whether the dataloaders should copy the Tensors into CUDA
                pinned memory (default False)
            use_gpu: True if training on GPUs, False otherwise
        """
        pass

    @abstractmethod
    def train_step(self, use_gpu, local_variables: Optional[Dict] = None) -> None:
        """
        Run a train step.

        This corresponds to training over one batch of data from the dataloaders.

        Args:
            use_gpu: True if training on GPUs, False otherwise
            local_variables: Local variables created in the function. Can be passed to
                custom :class:`ClassyHook`s.
        """
        pass

    def run_hooks(self, local_variables: Dict[str, Any], hook_function: str) -> None:
        """
        Helper function that runs a hook function for all the :class:`ClassyHook`s.

        Args:
            local_variables: Local variables created in :method:`train_step`
            hook_function: One of the hook functions in the :class:`ClassyHookFunctions`
                enum.
        """
        for hook in self.hooks:
            getattr(hook, hook_function)(self, local_variables)
