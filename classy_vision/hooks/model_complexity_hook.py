#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from classy_vision.generic.profiler import (
    ClassyProfilerNotImplementedError,
    compute_activations,
    compute_flops,
    count_params,
)
from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook


@register_hook("model_complexity")
class ModelComplexityHook(ClassyHook):
    """
    Logs the number of paramaters and forward pass FLOPs and activations of the model.
    """

    on_phase_start = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self) -> None:
        super().__init__()
        self.num_flops = None
        self.num_activations = None
        self.num_parameters = None

    def on_start(self, task) -> None:
        """Measure number of parameters, FLOPs and activations."""
        self.num_flops = 0
        self.num_activations = 0
        self.num_parameters = 0
        try:
            self.num_parameters = count_params(task.base_model)
            logging.info("Number of parameters in model: %d" % self.num_parameters)
            try:
                self.num_flops = compute_flops(
                    task.base_model,
                    input_shape=task.base_model.input_shape,
                    input_key=task.base_model.input_key
                    if hasattr(task.base_model, "input_key")
                    else None,
                )
                if self.num_flops is None:
                    logging.info("FLOPs for forward pass: skipped.")
                    self.num_flops = 0
                else:
                    logging.info(
                        "FLOPs for forward pass: %d MFLOPs"
                        % (float(self.num_flops) / 1e6)
                    )
            except ClassyProfilerNotImplementedError as e:
                logging.warning(f"Could not compute FLOPs for model forward pass: {e}")
            try:
                self.num_activations = compute_activations(
                    task.base_model,
                    input_shape=task.base_model.input_shape,
                    input_key=task.base_model.input_key
                    if hasattr(task.base_model, "input_key")
                    else None,
                )
                logging.info(f"Number of activations in model: {self.num_activations}")
            except ClassyProfilerNotImplementedError as e:
                logging.warning(
                    f"Could not compute activations for model forward pass: {e}"
                )
        except Exception:
            logging.info("Skipping complexity calculation: Unexpected error")
            logging.debug("Error trace for complexity calculation:", exc_info=True)

    def get_summary(self):
        return {
            "FLOPS(M)": float(self.num_flops) / 1e6
            if self.num_flops is not None
            else 0,
            "num_activations(M)": float(self.num_activations) / 1e6
            if self.num_activations is not None
            else 0,
            "num_parameters(M)": float(self.num_parameters) / 1e6
            if self.num_parameters is not None
            else 0,
        }
