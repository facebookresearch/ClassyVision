#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from classy_vision.generic.classy_trainer_common import run_hooks, train_step
from classy_vision.generic.distributed_util import barrier, is_distributed_training_run
from classy_vision.hooks import ClassyHook, ClassyHookFunctions
from classy_vision.state.classy_state import ClassyState


class ClassyTrainer:
    def __init__(self, hooks, use_gpu):
        assert isinstance(hooks, list)
        assert all(isinstance(hook, ClassyHook) for hook in hooks)

        self.hooks = hooks
        self.use_gpu = use_gpu

    def train(self, task):
        """
        Runs training phases, phases are generated from the config.
        """

        state = task.build_initial_state()
        assert isinstance(state, ClassyState)

        if self.use_gpu:
            state.criterion = state.criterion.cuda()

        if is_distributed_training_run():
            state.init_distributed_data_parallel_model()

        local_variables = {}
        run_hooks(state, local_variables, self.hooks, ClassyHookFunctions.on_start.name)

        while not state.done_training():
            state.advance_phase()

            # Start phase hooks
            run_hooks(
                state,
                local_variables,
                self.hooks,
                ClassyHookFunctions.on_phase_start.name,
            )
            while True:
                # Process next sample
                try:
                    state = train_step(state, self.hooks, self.use_gpu, local_variables)
                except StopIteration:
                    break

            barrier()
            run_hooks(
                state,
                local_variables,
                self.hooks,
                ClassyHookFunctions.on_phase_end.name,
            )

        run_hooks(state, local_variables, self.hooks, ClassyHookFunctions.on_end.name)
