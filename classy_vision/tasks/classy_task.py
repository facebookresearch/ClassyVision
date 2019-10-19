#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict, List

from classy_vision.criterions import ClassyCriterion, build_criterion
from classy_vision.dataset import ClassyDataset, build_dataset
from classy_vision.generic.distributed_util import (
    init_distributed_data_parallel_model,
    is_distributed_training_run,
)
from classy_vision.generic.util import copy_model_to_gpu, update_classy_state
from classy_vision.meters import build_meters
from classy_vision.models import ClassyVisionModel, build_model
from classy_vision.optim import ClassyOptimizer, build_optimizer


class ClassyTask(object):
    def __init__(self, num_phases: int):
        self.criterion = None
        self.datasets = {}
        self.meters = []
        self.num_phases = num_phases
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
        self.advance_to_next_phase = True
        self.num_updates = 0
        self.data_iterator = None
        self.num_samples_this_phase = 0
        self.losses = []

    def set_dataset(self, dataset: ClassyDataset, split: str):
        self.datasets[split] = dataset
        return self

    def set_optimizer(self, optimizer: ClassyOptimizer):
        self.optimizer = optimizer
        return self

    def set_criterion(self, criterion: ClassyCriterion):
        self.criterion = criterion
        return self

    def set_test_only(self, test_only: bool):
        self.test_only = test_only
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

    def set_model(self, model: ClassyVisionModel):
        self.base_model = model
        return self

    @classmethod
    def from_config(cls, config):
        optimizer_config = config["optimizer"]
        optimizer_config["num_epochs"] = config["num_phases"]

        datasets = {}
        splits = ["train", "test"]
        for split in splits:
            datasets[split] = build_dataset(config["dataset"][split])
        criterion = build_criterion(config["criterion"])
        test_only = config["test_only"]
        meters = build_meters(config.get("meters", {}))
        model = build_model(config["model"])
        # put model in eval mode in case any hooks modify model states, it'll
        # be reset to train mode before training
        model.eval()
        optimizer = build_optimizer(optimizer_config)

        task = (
            cls(num_phases=config["num_phases"])
            .set_criterion(criterion)
            .set_test_only(test_only)
            .set_model(model)
            .set_optimizer(optimizer)
            .set_meters(meters)
        )
        for split in splits:
            task.set_dataset(datasets[split], split)

        return task

    def get_config(self):
        return {
            "criterion": self.criterion._config_DO_NOT_USE,
            "dataset": {
                split: dataset._config_DO_NOT_USE
                for split, dataset in self.datasets.items()
            },
            "meters": [meter._config_DO_NOT_USE for meter in self.meters],
            "model": self.base_model._config_DO_NOT_USE,
            "num_phases": self.num_phases,
            "optimizer": self.optimizer._config_DO_NOT_USE,
            "test_only": self.test_only,
        }

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
            phases = [{"train": True} for _ in range(self.num_phases)]

            final_phases = []
            for phase in phases:
                final_phases.append(phase)
                final_phases.append({"train": False})
            return final_phases

        return [{"train": False} for _ in range(self.num_phases)]

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

    def set_checkpoint(self, checkpoint):
        assert (
            checkpoint is None or "classy_state_dict" in checkpoint
        ), "Checkpoint does not contain classy_state_dict"
        self.checkpoint = checkpoint

    def prepare(self, num_workers=0, pin_memory=False, use_gpu=False):
        """
        Prepares the task for training.
        """
        self.phases = self._build_phases()
        self.dataloaders = self.build_dataloaders(
            num_workers=num_workers, pin_memory=pin_memory
        )

        if use_gpu:
            self.criterion = self.criterion.cuda()
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

    def run_hooks(self, local_variables: Dict[str, Any], hook_function: str) -> None:
        """
        Helper function that runs hook_function for all the classy hooks.
        """
        for hook in self.hooks:
            getattr(hook, hook_function)(self, local_variables)

    @property
    def num_batches_per_phase(self):
        return len(self.data_iterator)

    @property
    def where(self):
        current_step = self.num_updates / self.get_global_batchsize()
        num_steps = self.get_total_training_phases() * self.num_batches_per_phase
        where = current_step / num_steps

        assert where >= 0 and where < 1, f"Invalid where: {where}"

        return where

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

    def step(self):
        pass

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
        if self.base_model.freeze_trunk:
            self.model.eval()
            for heads in self.base_model.get_heads().values():
                for h in heads.values():
                    h.train(phase["train"])
        else:
            self.base_model.train(phase["train"])

        if self.train and self.train_phase_idx >= 0:
            self.optimizer.update_schedule_on_epoch(self.where)

    def get_classy_state(self, deep_copy=False):
        """
        Returns a dictionary containing the state stored inside the object.

        If deep_copy is True (default False), creates a deep copy. Otherwise,
        the returned dict's attributes will be tied to the object's.
        """
        # NOTE: this does not return any task information since we are
        # planning on a refactor.
        classy_state_dict = {
            "train": self.train,
            "base_model": self.base_model.get_classy_state(),
            "meters": [meter.get_classy_state() for meter in self.meters],
            "optimizer": self.optimizer.get_classy_state(),
            "phase_idx": self.phase_idx,
            "train_phase_idx": self.train_phase_idx,
            "advance_to_next_phase": self.advance_to_next_phase,
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
