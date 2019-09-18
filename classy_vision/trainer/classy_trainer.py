#!/usr/bin/env python3

import torch
from classy_vision.generic.classy_trainer_common import run_hooks, train_step
from classy_vision.generic.distributed_util import barrier
from classy_vision.hooks.classy_hook import ClassyHook, ClassyHookFunctions
from classy_vision.state.classy_state import ClassyState


class ClassyTrainer:
    def run(self, state, hooks, use_gpu):
        """
        Runs training phases, phases are generated from the config.
        """

        # assertions:
        assert isinstance(state, ClassyState)
        if hooks is None:
            hooks = []
        assert isinstance(hooks, list)
        assert all(isinstance(hook, ClassyHook) for hook in hooks)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            state.init_distributed_data_parallel_model()

        local_variables = {}
        run_hooks(state, local_variables, hooks, ClassyHookFunctions.on_start.name)

        while not state.done_training():
            state.advance_phase()

            # Start phase hooks
            run_hooks(
                state, local_variables, hooks, ClassyHookFunctions.on_phase_start.name
            )
            while True:
                # Process next sample
                try:
                    state = train_step(state, hooks, use_gpu, local_variables)
                except StopIteration:
                    break

            barrier()
            run_hooks(
                state, local_variables, hooks, ClassyHookFunctions.on_phase_end.name
            )

        run_hooks(state, local_variables, hooks, ClassyHookFunctions.on_end.name)

        return state
