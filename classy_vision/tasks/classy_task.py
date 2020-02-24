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
    to train using a :class:`classy_vision.trainer.ClassyTrainer`.
    """

    def __init__(self) -> "ClassyTask":
        """
        Constructs a ClassyTask.
        """
        self.hooks = []

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyTask":
        """Instantiates a ClassyTask from a configuration.

        Args:
            config: A configuration for a ClassyTask.

        Returns:
            A ClassyTask instance.
        """
        raise NotImplementedError()

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
    def done_training(self) -> bool:
        """
        Tells if we are done training.

        Returns:
            A boolean telling if training is over.
        """
        pass

    @abstractmethod
    def get_classy_state(self, deep_copy: bool = False) -> Dict[str, Any]:
        """Get the state of the ClassyTask.

        The returned state is used for checkpointing.

        Args:
            deep_copy: If True, creates a deep copy of the state dict. Otherwise, the
                returned dict's state will be tied to the object's.

        Returns:
            A state dictionary containing the state of the task.
        """
        pass

    @abstractmethod
    def set_classy_state(self, state):
        """Set the state of the ClassyTask.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the task from a checkpoint.
        """
        pass

    @abstractmethod
    def prepare(
        self,
        num_dataloader_workers=0,
        pin_memory=False,
        use_gpu=False,
        dataloader_mp_context=None,
    ) -> None:
        """
        Prepares the task for training.

        Will be called by the :class:`classy_vision.trainer.ClassyTrainer` to
        prepare the task, before on_start is called.

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
                custom :class:`classy_vision.hooks.ClassyHook`.
        """
        pass

    @abstractmethod
    def on_start(self, local_variables):
        """
        Start training.

        Called by :class:`classy_vision.trainer.ClassyTrainer` before training starts.
        """
        pass

    @abstractmethod
    def on_phase_start(self, local_variables):
        """
        Epoch start.

        Called by :class:`classy_vision.trainer.ClassyTrainer` before each epoch starts.
        """
        pass

    @abstractmethod
    def on_phase_end(self, local_variables):
        """
        Epoch end.

        Called by :class:`classy_vision.trainer.ClassyTrainer` after each epoch ends.
        """
        pass

    @abstractmethod
    def on_end(self, local_variables):
        """
        Training end.

        Called by :class:`classy_vision.trainer.ClassyTrainer` after training ends.
        """
        pass

    @abstractmethod
    def eval_step(self, use_gpu, local_variables: Optional[Dict] = None) -> None:
        """
        Run an evaluation step.

        This corresponds to evaluating the model over one batch of data.

        Args:
            use_gpu: True if training on GPUs, False otherwise
            local_variables: Local variables created in the function. Can be passed to
                custom :class:`classy_vision.hooks.ClassyHook`.
        """
        pass

    def step(self, use_gpu, local_variables: Optional[Dict] = None) -> None:
        from classy_vision.hooks import ClassyHookFunctions

        if self.train:
            self.train_step(use_gpu, local_variables)
        else:
            self.eval_step(use_gpu, local_variables)

        self.run_hooks(local_variables, ClassyHookFunctions.on_step.name)

    def run_hooks(self, local_variables: Dict[str, Any], hook_function: str) -> None:
        """
        Helper function that runs a hook function for all the
        :class:`classy_vision.hooks.ClassyHook`.

        Args:
            local_variables: Local variables created in :func:`train_step`
            hook_function: One of the hook functions in the
            :class:`classy_vision.hooks.ClassyHookFunctions`
                enum.
        """
        for hook in self.hooks:
            getattr(hook, hook_function)(self, local_variables)
