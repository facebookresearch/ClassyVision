#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
from classy_vision.hooks import ClassyHook
from classy_vision.tasks import ClassyTask


class ExponentialMovingAverageModelHook(ClassyHook):
    """
    Hook which keeps a track of the exponential moving average (EMA) of the model's
    parameters and applies the EMA params to the model during the test phases.

    Saving the state in cpu will save gpu memory, but will make training slower since
    the model parameters will need to be moved to cpu before the averaging.

    Note:
        This hooks stores two additional copies of the model's parameters, which will
        increase memory usage significantly.
    """

    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(
        self, decay: float, consider_bn_buffers: bool = True, device: str = "cpu"
    ) -> None:
        """The constructor method of ExponentialMovingAverageModelHook.

        Args:
            decay: EMA decay factor, should be in [0, 1]. A decay of 0 corresponds to
                always using the latest value (no EMA) and a decay of 1 corresponds to
                not updating weights after initialization.
            consider_bn_buffers: Whether to apply EMA to batch norm buffers
            device: Device to store the model state.
        """
        super().__init__()
        assert 0 <= decay <= 1, "Decay should be between 0 and 1"
        assert device in ["cpu", "gpu"], "Device should be one of cpu or gpu"
        self.decay: int = decay
        self.consider_bn_buffers = consider_bn_buffers
        self.device = "cuda" if device == "gpu" else "cpu"
        self.state.model_state = {}
        self.state.ema_model_state = {}
        logging.info(
            f"{self.__class__.__name__} initialized with a decay of "
            f"{decay} on device {device}"
        )

    def get_model_state_iterator(self, model: nn.Module) -> Iterable[Tuple[str, Any]]:
        """Get an iterator over the model state to apply EMA to."""
        iterable = model.named_parameters()
        if self.consider_bn_buffers:
            # also add batch norm buffers to the list of state params to iterate over
            buffers_iterable = (
                (f"{module_name}_buffer_{name}", buffer)
                for module_name, module in model.named_modules()
                for name, buffer in module.named_buffers()
                if isinstance(
                    module,
                    (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
                )
            )
            iterable = itertools.chain(iterable, buffers_iterable)
        return iterable

    def _save_current_model_state(self, model: nn.Module, model_state: Dict[str, Any]):
        """Copy the model's state to the provided dict."""
        for name, param in self.get_model_state_iterator(model):
            model_state[name] = param.detach().clone().to(device=self.device)

    def on_start(self, task: ClassyTask, local_variables: Dict[str, Any]) -> None:
        if self.state.model_state:
            # loaded state from checkpoint, do not re-initialize, only move the state
            # to the right device
            for name in self.state.model_state:
                self.state.model_state[name] = self.state.model_state[name].to(
                    device=self.device
                )
                self.state.ema_model_state[name] = self.state.ema_model_state[name].to(
                    device=self.device
                )
            return
        self._save_current_model_state(task.base_model, self.state.model_state)
        self._save_current_model_state(task.base_model, self.state.ema_model_state)

    def on_phase_start(self, task: ClassyTask, local_variables: Dict[str, Any]) -> None:
        # restore the right state depending on the phase type
        self.set_model_state(task, use_ema=not task.train)

    def on_phase_end(self, task: ClassyTask, local_variables: Dict[str, Any]) -> None:
        if task.train:
            # save the current model state since this will be overwritten by the ema
            # state in the test phase
            self._save_current_model_state(task.base_model, self.state.model_state)

    def on_step(self, task: ClassyTask, local_variables: Dict[str, Any]) -> None:
        if not task.train:
            return

        with torch.no_grad():
            for name, param in self.get_model_state_iterator(task.base_model):
                self.state.ema_model_state[
                    name
                ] = self.decay * self.state.ema_model_state[name] + (
                    1 - self.decay
                ) * param.to(
                    device=self.device
                )

    def set_model_state(self, task: ClassyTask, use_ema: bool) -> None:
        """
        Depending on use_ema, set the appropriate state for the model.
        """
        model_state = self.state.ema_model_state if use_ema else self.state.model_state
        with torch.no_grad():
            for name, param in self.get_model_state_iterator(task.base_model):
                param.copy_(model_state[name])
