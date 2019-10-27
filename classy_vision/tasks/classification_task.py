#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import List, Union

import torch
from classy_vision.dataset import ClassyDataset, build_dataset
from classy_vision.generic.distributed_util import (
    all_reduce_mean,
    init_distributed_data_parallel_model,
    is_distributed_training_run,
)
from classy_vision.generic.perf_stats import PerfTimer
from classy_vision.generic.util import (
    copy_model_to_gpu,
    recursive_copy_to_gpu,
    update_classy_state,
)
from classy_vision.losses import ClassyLoss, build_loss
from classy_vision.meters import build_meters
from classy_vision.models import ClassyModel, build_model
from classy_vision.optim import ClassyOptimizer, build_optimizer

from . import register_task
from .classy_task import ClassyTask


@register_task("classification_task")
class ClassificationTask(ClassyTask):
    def __init__(self):
        super().__init__()

        self.loss = None
        self.datasets = {}
        self.meters = []
        self.num_epochs = 1
        self.test_only = False
        self.base_model = None
        self.optimizer = None
        self.checkpoint = None
        self.phases = []
        self.hooks = []
        self.train = True
        self.distributed_model = None
        self.phase_idx = -1
        self.train_phase_idx = -1
        self.num_updates = 0
        self.data_iterator = None
        self.num_samples_this_phase = 0
        self.losses = []

    def set_checkpoint(self, checkpoint):
        assert (
            checkpoint is None or "classy_state_dict" in checkpoint
        ), "Checkpoint does not contain classy_state_dict"
        self.checkpoint = checkpoint

    def set_num_epochs(self, num_epochs: Union[int, float]):
        self.num_epochs = num_epochs
        return self

    def set_dataset(self, dataset: ClassyDataset, split: str):
        self.datasets[split] = dataset
        return self

    def set_optimizer(self, optimizer: ClassyOptimizer):
        self.optimizer = optimizer
        return self

    def set_loss(self, loss: ClassyLoss):
        self.loss = loss
        return self

    def set_meters(self, meters: List["ClassyMeter"]):
        self.meters = meters
        return self

    def set_hooks(self, hooks: List["ClassyHook"]):
        from classy_vision.hooks import ClassyHook

        assert isinstance(hooks, list)
        assert all(isinstance(hook, ClassyHook) for hook in hooks)
        assert len({hook.name() for hook in hooks}) == len(
            hooks
        ), "Cannot have repeated hooks of the same class"

        self.hooks = hooks
        return self

    def set_model(self, model: ClassyModel):
        self.base_model = model
        return self

    def set_test_only(self, test_only: bool):
        self.test_only = test_only
        return self

    @classmethod
    def from_config(cls, config):
        optimizer_config = config["optimizer"]
        optimizer_config["num_epochs"] = config["num_epochs"]

        datasets = {}
        splits = ["train", "test"]
        for split in splits:
            datasets[split] = build_dataset(config["dataset"][split])
        loss = build_loss(config["loss"])
        test_only = config["test_only"]
        meters = build_meters(config.get("meters", {}))
        model = build_model(config["model"])
        # put model in eval mode in case any hooks modify model states, it'll
        # be reset to train mode before training
        model.eval()
        optimizer = build_optimizer(optimizer_config)

        task = (
            cls()
            .set_num_epochs(config["num_epochs"])
            .set_loss(loss)
            .set_test_only(test_only)
            .set_model(model)
            .set_optimizer(optimizer)
            .set_meters(meters)
        )
        for split in splits:
            task.set_dataset(datasets[split], split)

        return task

    @property
    def num_batches_per_phase(self):
        return len(self.data_iterator)

    @property
    def model(self):
        return (
            self.distributed_model if is_distributed_training_run() else self.base_model
        )

    @property
    def phase_type(self):
        return "train" if self.train else "test"

    @property
    def eval_phase_idx(self):
        return self.phase_idx - self.train_phase_idx - 1

    def get_data_iterator(self):
        return self.data_iterator

    def get_config(self):
        return {
            "loss": self.loss._config_DO_NOT_USE,
            "dataset": {
                split: dataset._config_DO_NOT_USE
                for split, dataset in self.datasets.items()
            },
            "meters": [meter._config_DO_NOT_USE for meter in self.meters],
            "model": self.base_model._config_DO_NOT_USE,
            "num_epochs": self.num_epochs,
            "optimizer": self.optimizer._config_DO_NOT_USE,
            "test_only": self.test_only,
        }

    def get_total_training_phases(self):
        """
        Returns the total number of "train" phases in the list of execution
        phases
        """
        num_training_phases = 0
        for phase in self.phases:
            if phase["train"] is True:
                num_training_phases += 1
        return num_training_phases

    def _build_phases(self):
        """
        Returns list of phases from config.  These phases will look like:
        {
          train: is this a train or test phase?
          optimizer: optimizer settings
        }

        If this is a test only run, then only test phases will be
        generated, if this is a training run, then x phases = x train
        phases + x test phases, interleaved.
        """
        if not self.test_only:
            phases = [{"train": True} for _ in range(self.num_epochs)]

            final_phases = []
            for phase in phases:
                final_phases.append(phase)
                final_phases.append({"train": False})
            return final_phases

        return [{"train": False} for _ in range(self.num_epochs)]

    def build_dataloader(self, split, num_workers, pin_memory, **kwargs):
        return self.datasets[split].iterator(
            num_workers=num_workers, pin_memory=pin_memory, **kwargs
        )

    def build_dataloaders(self, num_workers, pin_memory, **kwargs):
        return {
            split: self.build_dataloader(
                split, num_workers=num_workers, pin_memory=pin_memory, **kwargs
            )
            for split in self.datasets.keys()
        }

    def prepare(self, num_dataloader_workers=0, pin_memory=False, use_gpu=False):
        self.phases = self._build_phases()
        self.dataloaders = self.build_dataloaders(
            num_workers=num_dataloader_workers, pin_memory=pin_memory
        )

        if use_gpu:
            self.loss = self.loss.cuda()
            self.base_model = copy_model_to_gpu(self.base_model)

        # initialize the pytorch optimizer now since the model has been moved to
        # the appropriate device
        self.optimizer.init_pytorch_optimizer(self.base_model)

        classy_state_dict = (
            None
            if self.checkpoint is None
            else self.checkpoint.get("classy_state_dict")
        )

        if classy_state_dict is not None:
            state_load_success = update_classy_state(self, classy_state_dict)
            assert (
                state_load_success
            ), "Update classy state from checkpoint was unsuccessful."

    def init_distributed_data_parallel_model(self):
        assert (
            self.distributed_model is None
        ), "init_ddp_non_elastic must only be called once"

        self.distributed_model = init_distributed_data_parallel_model(self.base_model)

    @property
    def where(self):
        current_step = self.num_updates / self.get_global_batchsize()
        num_steps = self.get_total_training_phases() * self.num_batches_per_phase
        where = current_step / num_steps

        assert where >= 0 and where < 1, f"Invalid where: {where}"

        return where

    def get_classy_state(self, deep_copy=False):
        classy_state_dict = {
            "train": self.train,
            "base_model": self.base_model.get_classy_state(),
            "meters": [meter.get_classy_state() for meter in self.meters],
            "optimizer": self.optimizer.get_classy_state(),
            "phase_idx": self.phase_idx,
            "train_phase_idx": self.train_phase_idx,
            "num_updates": self.num_updates,
            "num_samples_this_phase": self.num_samples_this_phase,
            "losses": self.losses,
            "hooks": {hook.name(): hook.state_dict() for hook in self.hooks},
        }
        if deep_copy:
            classy_state_dict = copy.deepcopy(classy_state_dict)
        return classy_state_dict

    def set_classy_state(self, state):
        self.train = state["train"]
        self.base_model.set_classy_state(state["base_model"])
        for meter, meter_state in zip(self.meters, state["meters"]):
            meter.set_classy_state(meter_state)
        self.optimizer.set_classy_state(state["optimizer"])
        self.phase_idx = state["phase_idx"]
        self.train_phase_idx = state["train_phase_idx"]
        self.num_updates = state["num_updates"]
        self.num_samples_this_phase = state["num_samples_this_phase"]
        self.losses = state["losses"]
        for hook in self.hooks:
            # we still want to be able to run when new hooks are added or old
            # hooks are removed
            if hook.name() in state["hooks"]:
                hook.load_state_dict(state["hooks"][hook.name()])
            else:
                logging.warn(f"No state found for hook: {hook.name()}")
        # TODO (mannatsingh): Figure out how to set the state of the dataloaders
        # Re-build dataloader & re-create iterator.
        self._recreate_data_loader_from_dataset()
        self._reshuffle_data()
        self.create_data_iterator()
        # Set up pytorch module in train vs eval mode, update optimizer.
        self._set_model_train_mode()

    def train_step(self, use_gpu, local_variables=None):
        from classy_vision.hooks import ClassyHookFunctions

        if local_variables is None:
            local_variables = {}

        # We'll time train_step and some of its sections, and accumulate values
        # into perf_stats if it were defined in local_variables:
        perf_stats = local_variables.get("perf_stats", None)
        timer_train_step = PerfTimer("train_step_total", perf_stats)
        timer_train_step.start()

        # Process next sample
        with PerfTimer("read_sample", perf_stats):
            sample = next(self.get_data_iterator())
            local_variables["sample"] = sample

            assert (
                isinstance(local_variables["sample"], dict)
                and "input" in local_variables["sample"]
                and "target" in local_variables["sample"]
            ), (
                f"Returned sample [{sample}] is not a map with 'input' and"
                + "'target' keys"
            )

        self.run_hooks(local_variables, ClassyHookFunctions.on_sample.name)

        # Copy sample to GPU
        local_variables["target"] = local_variables["sample"]["target"]
        if use_gpu:
            for key, value in local_variables["sample"].items():
                local_variables["sample"][key] = recursive_copy_to_gpu(
                    value, non_blocking=True
                )

        # Only need gradients during training
        context = torch.enable_grad() if self.train else torch.no_grad()
        with context:
            # Forward pass
            with PerfTimer("forward", perf_stats):
                local_variables["output"] = self.model(
                    local_variables["sample"]["input"]
                )

            self.run_hooks(local_variables, ClassyHookFunctions.on_forward.name)

            model_output = local_variables["output"]
            target = local_variables["sample"]["target"]
            local_variables["local_loss"] = self.loss(model_output, target)

            # NOTE: This performs an all_reduce_mean() on the losses across the
            # replicas.  The reduce should ideally be weighted by the length of
            # the targets on each replica. This will only be an issue when
            # there are dummy samples present (once an epoch) and will only
            # impact the loss reporting (slightly).
            with PerfTimer("loss_allreduce", perf_stats):
                local_variables["loss"] = local_variables["local_loss"].detach().clone()
                local_variables["loss"] = all_reduce_mean(local_variables["loss"])

            self.losses.append(
                local_variables["loss"].data.cpu().item()
                * local_variables["target"].size(0)
            )

            self.run_hooks(local_variables, ClassyHookFunctions.on_loss.name)

            model_output_cpu = model_output.cpu() if use_gpu else model_output

            # Update meters
            with PerfTimer("meters_update", perf_stats):
                for meter in self.meters:
                    meter.update(
                        model_output_cpu, target.detach().cpu(), is_train=self.train
                    )

        num_samples_in_step = self.get_global_batchsize()
        self.num_samples_this_phase += num_samples_in_step

        # For training phases, run backwards pass / update optimizer
        if self.train:
            with PerfTimer("backward", perf_stats):
                self.optimizer.backward(local_variables["local_loss"])

            self.run_hooks(local_variables, ClassyHookFunctions.on_backward.name)

            self.optimizer.update_schedule_on_step(self.where)
            with PerfTimer("optimizer_step", perf_stats):
                self.optimizer.step()

            self.run_hooks(local_variables, ClassyHookFunctions.on_update.name)

            self.num_updates += num_samples_in_step

        timer_train_step.stop()
        timer_train_step.record()

    def advance_phase(self):
        # Reset meters for next phase / epoch
        for meter in self.meters:
            meter.reset()

        # Reset loss history for next epoch
        self.losses = []

        # Setup new phase
        self.num_samples_this_phase = 0
        self.phase_idx += 1
        phase = self.phases[self.phase_idx]
        self.train = True if phase["train"] else False
        if self.train:
            self.train_phase_idx += 1

        # Re-build dataloader & re-create iterator anytime membership changes.
        self._recreate_data_loader_from_dataset()
        self._reshuffle_data()
        self.create_data_iterator()
        # Set up pytorch module in train vs eval mode, update optimizer.
        self._set_model_train_mode()

    def done_training(self):
        return self.phase_idx + 1 >= len(self.phases)

    def _recreate_data_loader_from_dataset(self, phase_type=None):
        """
        This utility is invoked to re-create the data loader object
        for the current phase of execution, using the existing dataset.
        This is sufficient when advancing phases.
        """
        if phase_type is None:
            phase_type = self.phase_type

        logging.info("Recreating data loader for new phase")
        num_workers = 0
        if hasattr(self.dataloaders[phase_type], "num_workers"):
            num_workers = self.dataloaders[phase_type].num_workers
        pin_memory = False
        if hasattr(self.dataloaders[phase_type], "pin_memory"):
            pin_memory = self.dataloaders[phase_type].pin_memory
        if phase_type == "test":
            current_phase_id = 0
        else:
            current_phase_id = max(self.train_phase_idx, 0)

        self.dataloaders[phase_type] = self.build_dataloader(
            split=phase_type,
            num_workers=num_workers,
            pin_memory=pin_memory,
            current_phase_id=current_phase_id,
        )

    def _reshuffle_data(self):
        # (Re-)Shuffle data if needed
        if hasattr(self.dataloaders[self.phase_type].dataset, "do_shuffle"):
            self.dataloaders[self.phase_type].dataset.do_shuffle(
                epoch_num=self.phase_idx
            )
            logging.info("Data shuffled.")

    def create_data_iterator(self):
        # Delete iterator explicitly so that all dataloader processes
        # are cleaned up.
        del self.data_iterator
        self.data_iterator = iter(self.dataloaders[self.phase_type])

    def _set_model_train_mode(self):
        phase = self.phases[self.phase_idx]
        self.base_model.train(phase["train"])

        if self.train and self.train_phase_idx >= 0:
            self.optimizer.update_schedule_on_epoch(self.where)

    # Functions below should be better abstracted into the dataloader
    # abstraction
    def get_batchsize_per_replica(self):
        # TODO(T47573564) - cleaner abstraction
        return self.dataloaders[self.phase_type].dataset.get_batchsize_per_replica()

    def get_global_batchsize(self):
        return self.dataloaders[self.phase_type].dataset.get_global_batchsize()

    def get_total_samples_trained_this_phase(self):
        # TODO(T47573564) - cleaner abstraction
        # TODO(T47387605) - instead of get_world_size, we need the max world
        # size for elasticity to match parity with Uru and other systems,
        # although DPP will solve this by dynamically re-sharding.
        return self.num_samples_this_phase
