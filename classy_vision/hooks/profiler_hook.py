#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

from classy_vision import tasks
from classy_vision.generic.profiler import profile, summarize_profiler_info
from classy_vision.hooks.classy_hook import ClassyHook


class ProfilerHook(ClassyHook):
    """
    Hook to profile a model.
    """

    on_rendezvous = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_sample = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def on_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """
        Profile the forward pass.
        """
        logging.info("Profiling forward pass...")
        batchsize_per_replica = getattr(
            task.dataloaders[task.phase_type].dataset, "batchsize_per_replica", 1
        )
        input_shape = task.base_model.input_shape
        p = profile(
            task.model,
            batchsize_per_replica=batchsize_per_replica,
            input_shape=input_shape,
        )
        logging.info(summarize_profiler_info(p))
