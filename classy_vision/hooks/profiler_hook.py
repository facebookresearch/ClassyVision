#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

from classy_vision.generic.profiler import profile, summarize_profiler_info
from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook


@register_hook("profiler")
class ProfilerHook(ClassyHook):
    """
    Hook to profile a model and to show model runtime information, such as
        the time breakdown in milliseconds of forward/backward pass.
    """

    on_phase_start = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def on_start(self, task) -> None:
        """Profile the forward pass."""
        logging.info("Profiling forward pass...")
        batchsize_per_replica = task.get_batchsize_per_replica()
        input_shape = task.base_model.input_shape
        p = profile(
            task.model,
            batchsize_per_replica=batchsize_per_replica,
            input_shape=input_shape,
            input_key=getattr(task.base_model, "input_key", None),
        )
        logging.info(summarize_profiler_info(p))
