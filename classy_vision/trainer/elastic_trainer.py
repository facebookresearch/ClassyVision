#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import os
from typing import Any, Optional

import numpy
import torch
import torchelastic
import torchelastic.distributed as dist
from classy_vision.generic.distributed_util import set_cpu_device, set_cuda_device_index
from classy_vision.generic.util import get_checkpoint_dict
from classy_vision.hooks import ClassyHookFunctions
from classy_vision.tasks import ClassyTask
from classy_vision.trainer import ClassyTrainer
from torchelastic.worker_stats import WorkerStats


log = logging.getLogger(__name__)


class ElasticTrainer(ClassyTrainer):
    def __init__(
        self,
        use_gpu,
        num_dataloader_workers,
        elastic_coordinator,
        input_args,
        local_rank,
        dataloader_mp_context=None,
    ):
        super().__init__(
            use_gpu=use_gpu,
            num_dataloader_workers=num_dataloader_workers,
            dataloader_mp_context=dataloader_mp_context,
        )
        pid = os.getpid()
        if use_gpu:
            set_cuda_device_index(local_rank)
            device_idx = torch.cuda.current_device()
            log.info(f"initialized worker {local_rank} (pid={pid}, gpu={device_idx})")
            device_properties = torch.cuda.get_device_properties(device_idx)
            log.info(f"gpu device properties: {device_properties}")
        else:
            # cpu
            set_cpu_device()
            log.info(f"initialized worker {local_rank} (pid={pid}, cpu)")

        self.elastic_coordinator = elastic_coordinator
        self.input_args = input_args

    def train(self, task):
        """
        Runs training phases, phases are generated from the config.
        """

        assert isinstance(task, ClassyTask)
        pin_memory = self.use_gpu and torch.cuda.device_count() > 1

        task.prepare(
            num_dataloader_workers=self.num_dataloader_workers,
            pin_memory=pin_memory,
            use_gpu=self.use_gpu,
            dataloader_mp_context=self.dataloader_mp_context,
        )
        state = self._ClassyElasticState(task, self.input_args)

        local_variables = {}

        state.advance_to_next_phase = True

        def elastic_train_step(orig_state):
            if state.run_start_hooks:
                # need this to ensure we don't run the on_start hooks every time
                # a trainer starts
                state.task.on_start(local_variables)
                state.run_start_hooks = False
                return state, self._ClassyWorkerStats(None)

            return self._run_step(orig_state, local_variables, self.use_gpu)

        torchelastic.train(self.elastic_coordinator, elastic_train_step, state)

        task.on_end(local_variables)

    def _run_step(self, state, local_variables, use_gpu):
        # Check for training complete but only terminate when the last phase is done
        if state.task.done_training() and state.advance_to_next_phase:
            raise StopIteration

        if state.advance_to_next_phase:
            self.elastic_coordinator.barrier()
            self.elastic_coordinator._log_event("on_phase_start")
            state.task.on_phase_start(local_variables)

            state.advance_to_next_phase = False

        # Process one train step
        try:
            if state.skip_current_phase:
                state.advance_to_next_phase = True
                state.skip_current_phase = False  # Reset flag
            else:
                state.task.step(use_gpu, local_variables)
        except StopIteration:
            state.advance_to_next_phase = True

        if state.advance_to_next_phase:
            self.elastic_coordinator.barrier()
            state.task.on_phase_end(local_variables)

        progress_rate = None  # using None to signal 'unknown'
        perf_stats = local_variables.get("perf_stats", None)
        if perf_stats is not None:
            batch_time = perf_stats._cuda_stats["train_step_total"].smoothed_value
            if batch_time is not None and batch_time > 0.0:
                # rate = number of mini-batches per second
                progress_rate = 1.0 / batch_time

        progress_stats = self._ClassyWorkerStats(progress_rate)
        return state, progress_stats

    class _ClassyWorkerStats(WorkerStats):
        """
        ClassyVision-specific implementation of WorkerStats,
        which is used by torchelastic train_loop
        to detect (and correct stragglers), or other progress-impeding issues.
        """

        def __init__(self, progress_rate):
            self.progress_rate = progress_rate

        def get_progress_rate(self) -> Optional[float]:
            return self.progress_rate

    class _ClassyElasticState(torchelastic.State):
        """
        Rollback is disabled on this state since currently, data loaders are
        too expensive to snapshot on every train_step
        """

        def __init__(self, task: ClassyTask, input_args: Any):
            # WARNING: Make sure to add any members here to self.save() and self.load()
            self.task = task
            self.input_args = input_args if input_args else {}
            self.advance_to_next_phase = True
            self.skip_current_phase = False
            self.snapshot = None
            # run_start_hooks is used to determine if the on_start hooks should be run,
            # which should happen at the beginning of training. After that, this is set
            # to False.
            self.run_start_hooks = True

        def broadcast_state(self, rank, src_rank):
            data = None
            if rank == src_rank:
                save_stream = io.BytesIO()
                self.save(save_stream)
                # Note: save_stream.getbuffer() will return a memoryview, which
                # cannot be convert to a tensor, need convert it to np array first
                data = numpy.asarray(save_stream.getbuffer())
            data = dist.broadcast_binary(data, src_rank)
            load_stream = io.BytesIO(data)
            self.load(load_stream)

        def sync(self, world_size, rank):
            self._recreate_ddp_model()

            # Figure out which trainer has the most up-to-date data, and
            # use that trainer to broadcast task to all others.
            src_rank = self._compute_most_tenured_rank(rank)
            self.broadcast_state(rank, src_rank)

            # Current on-box data loaders don't support recovery in the middle of
            # a phase and since we don't rollback the model, re-training is
            # worse than losing data so we're skipping rest of the phase.
            #
            # Also we can't just set advance_to_next_phase to True here as it
            # will cause on_phase_end() hooks to not run.
            # We also only skip the current phase if this isn't the first time
            # calling sync from the PET train_loop. We'll need to reconsider this
            # if the PET train_loop changes. advance_to_next_phase is already
            # synced from rest of the trainers at this point.
            if not self.advance_to_next_phase:
                self.skip_current_phase = True

            logging.warning(
                "RANK {}: now we all have {} updates and latest task".format(
                    rank, self.task.num_updates
                )
            )

            # Re-build dataloader, dataset, and iterator anytime membership
            # changes. When world_size potentially changes (e.g. re-rendezvous), we
            # need to re-create both the dataset and dataloader objects because we
            # create a ShardDataset based on the world size at the time of
            # construction.
            # TODO (T55691442): Figure out how to solve re-sharding without
            # rebuilding the datasets. sync() only works correctly without elasticity
            # currently.
            for phase_type in self.task.datasets.keys():
                self.task._recreate_data_loader_from_dataset(phase_type)
            self.task.create_data_iterator()
            # Set up pytorch module in train vs eval mode, update optimizer.
            self.task._set_model_train_mode()

        def should_save_checkpoint(self, rank):
            # should_save_checkpoint need to return same value for all trainers
            # we take checkpoint when a phase completed
            # TODO add test coverage for this

            # currently for typical imagenet resnet model checkpointing take 15 seconds
            # consider the cost it is not very necessary to do checkpoint for test phase
            return self.task.train and self.advance_to_next_phase

        def capture_snapshot(self):
            # Save snapshot at phase boundary. This will make sure no data-loss
            # when a failure happens. We will support fine-grade recovery once
            # fault tolerate data loader is ready
            if self.task.train and self.advance_to_next_phase:
                stream = io.BytesIO()
                self.save(stream)
                # save snapshot and return it every train step this make sure
                # we always has a good state to recover.
                self.snapshot = stream.getbuffer()
                logging.info(
                    "Take snapshot at updates {}".format(self.task.num_updates)
                )
            return self.snapshot

        def apply_snapshot(self, capture_snapshot) -> None:
            with io.BytesIO(capture_snapshot) as stream:
                self.load(stream)
                logging.info(
                    "Snapshot applied, now we are at updates {}".format(
                        self.task.num_updates
                    )
                )

        def save(self, stream):
            checkpoint_state = get_checkpoint_dict(self.task, self.input_args)
            checkpoint_state["advance_to_next_phase"] = self.advance_to_next_phase
            checkpoint_state["skip_current_phase"] = self.skip_current_phase
            checkpoint_state["run_start_hooks"] = self.run_start_hooks
            torch.save(checkpoint_state, stream)

        def load(self, stream):
            checkpoint_state = torch.load(stream, map_location=torch.device("cpu"))
            state = checkpoint_state["classy_state_dict"]
            self.task.set_classy_state(state)
            if "advance_to_next_phase" in checkpoint_state:
                self.advance_to_next_phase = checkpoint_state["advance_to_next_phase"]
            if "skip_current_phase" in checkpoint_state:
                self.skip_current_phase = checkpoint_state["skip_current_phase"]
            if "run_start_hooks" in checkpoint_state:
                self.run_start_hooks = checkpoint_state["run_start_hooks"]

        def _recreate_ddp_model(self):
            # Delete & re-create the DDP module wrapper. This is required because
            # each instance of DDP is tied to a specific process group, and
            # any time the set of workers in PET changes, we create a new
            # process group, so the old DDP wrapper is invalid.
            # TODO: does calling del here invoke C++ destructor if it's the last
            # reference? Or is assigning None sufficient?
            del self.task.distributed_model
            self.task.distributed_model = None
            self.task.init_distributed_data_parallel_model()

        def _compute_most_tenured_rank(self, rank):
            logging.warning(
                "RANK {}: syncing, I have {} updates".format(
                    rank, self.task.num_updates
                )
            )
            # Propagate state to new trainer processes.
            # First, figure out which process has a copy of the most recent
            # state by getting a copy of everybody's iteration counter.
            max_rank, max_num_updates = dist.all_gather_return_max_long(
                self.task.num_updates
            )

            logging.warning(
                "RANK {}: rank {} has the most updates {}".format(
                    rank, max_rank, max_num_updates
                )
            )

            return max_rank
