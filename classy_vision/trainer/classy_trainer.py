#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from classy_vision.generic.classy_trainer_common import train_step
from classy_vision.generic.distributed_util import barrier, is_distributed_training_run
from classy_vision.generic.util import copy_model_to_gpu
from classy_vision.hooks import ClassyHookFunctions
from classy_vision.state.classy_state import ClassyState


class ClassyTrainer:
    def __init__(self, use_gpu, num_workers=0):
        self.use_gpu = use_gpu
        self.num_workers = num_workers

    def train(self, task):
        """
        Runs training phases, phases are generated from the config.
        """

        pin_memory = self.use_gpu and torch.cuda.device_count() > 1
        state = task.build_initial_state(
            num_workers=self.num_workers, pin_memory=pin_memory
        )
        assert isinstance(state, ClassyState)

        if self.use_gpu:
            state.criterion = state.criterion.cuda()
            state.base_model = copy_model_to_gpu(state.base_model)

        if is_distributed_training_run():
            state.init_distributed_data_parallel_model()

        local_variables = {}
        state.run_hooks(local_variables, ClassyHookFunctions.on_start.name)

        while not state.done_training():
            state.advance_phase()

            # Start phase hooks
            state.run_hooks(local_variables, ClassyHookFunctions.on_phase_start.name)
            while True:
                # Process next sample
                try:
                    state = train_step(state, self.use_gpu, local_variables)
                except StopIteration:
                    break

            barrier()
            state.run_hooks(local_variables, ClassyHookFunctions.on_phase_end.name)

        state.run_hooks(local_variables, ClassyHookFunctions.on_end.name)
