#!/usr/bin/env python3

import logging
from typing import Any, Dict

from classy_vision.generic.profiler import profile, summarize_profiler_info
from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.state.classy_state import ClassyState


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

    def on_start(self, state: ClassyState, local_variables: Dict[str, Any]) -> None:
        """
        Profile the forward pass.
        """
        logging.info("Profiling forward pass...")
        batchsize_per_replica = getattr(
            state.dataloaders[state.phase_type].dataset, "batchsize_per_replica", 1
        )
        input_shape = state.base_model.input_shape
        p = profile(
            state.model,
            batchsize_per_replica=batchsize_per_replica,
            input_shape=input_shape,
        )
        logging.info(summarize_profiler_info(p))
