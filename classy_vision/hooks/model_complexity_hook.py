#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

from classy_vision import tasks
from classy_vision.generic.profiler import (
    compute_activations,
    compute_flops,
    count_params,
)
from classy_vision.hooks.classy_hook import ClassyHook


class ModelComplexityHook(ClassyHook):
    """
    Logs the number of paramaters and forward pass FLOPs and activations of the model.
    """

    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def on_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
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
            except NotImplementedError:
                logging.warning(
                    """Model contains unsupported modules:
                Could not compute FLOPs for model forward pass. Exception:""",
                    exc_info=True,
                )
            try:
                self.num_activations = compute_activations(
                    task.base_model,
                    input_shape=task.base_model.input_shape,
                    input_key=task.base_model.input_key
                    if hasattr(task.base_model, "input_key")
                    else None,
                )
                logging.info(f"Number of activations in model: {self.num_activations}")
            except NotImplementedError:
                logging.info(
                    "Model does not implement input_shape. Skipping "
                    "activation calculation."
                )
        except Exception:
            logging.exception("Unexpected failure estimating model complexity.")

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
