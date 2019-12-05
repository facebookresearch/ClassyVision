#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

from classy_vision import tasks
from classy_vision.generic.profiler import compute_flops, count_params
from classy_vision.hooks.classy_hook import ClassyHook


class ModelComplexityHook(ClassyHook):
    """
    Logs the number of paramaters and forward pass FLOPs of the model.
    """

    on_rendezvous = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_sample = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def on_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Measure number of parameters and number of FLOPs."""
        try:
            num_flops = compute_flops(
                task.base_model,
                input_shape=task.base_model.input_shape,
                input_key=task.base_model.input_key
                if hasattr(task.base_model, "input_key")
                else None,
            )
            if num_flops is None:
                logging.info("FLOPs for forward pass: skipped.")
            else:
                logging.info(
                    "FLOPs for forward pass: %d MFLOPs" % (float(num_flops) / 1e6)
                )
        except NotImplementedError:
            logging.warning(
                """Model contains unsupported modules:
            Could not compute FLOPs for model forward pass. Exception:""",
                exc_info=True,
            )
        logging.info(
            "Number of parameters in model: %d" % count_params(task.base_model)
        )
