#!/usr/bin/env python3

import copy
import logging

import torch
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    get_world_size,
    init_distributed_data_parallel_model,
)
from torch.utils.data import IterableDataset


class ClassyState:
    """
    Simple class that maintains the state of a trainer / tester.

    Initializer simply assigns provided stateful objects to the
    associated member functions. There are a few helper functions for
    dealing with the member objects.
    """

    def __init__(
        self,
        task,
        phases,
        train,  # indicates whether it is training or testing
        dataloaders,
        base_model,
        criterion,
        meters,
        optimizer,
        phase_idx=-1,
        train_phase_idx=-1,
        advance_to_next_phase=True,
        num_updates=0,
        num_samples_this_phase=0,
        losses=None,
    ):
        self.task = task
        self.phases = phases
        self.train = train
        self.dataloaders = dataloaders
        self.base_model = base_model
        self.distributed_model = None
        self.criterion = criterion
        self.meters = meters
        self.optimizer = optimizer
        self.num_updates = 0
        self.data_iterator = None
        self.phase_idx = phase_idx
        # TODO: we should probably not use the phase index to drive LR, but
        # instead use sample count or true epoch count?
        self.train_phase_idx = train_phase_idx
        self.advance_to_next_phase = advance_to_next_phase
        self.num_updates = num_updates
        self.num_samples_this_phase = num_samples_this_phase
        if losses is None:
            losses = []
        assert isinstance(losses, list), "losses should be a list"
        self.losses = losses

    def init_distributed_data_parallel_model(self):
        assert (
            self.distributed_model is None
        ), "init_ddp_non_elastic must only be called once"

        self.distributed_model = init_distributed_data_parallel_model(self.base_model)

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
            self.distributed_model
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else self.base_model
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
        return getattr(
            self.dataloaders[self.phase_type].dataset, "batchsize_per_replica", 1
        )

    def get_global_batchsize(self):
        return self.get_batchsize_per_replica() * get_world_size()

    def get_total_samples_trained_this_phase(self):
        # TODO(T47573564) - cleaner abstraction
        # TODO(T47387605) - instead of get_world_size, we need the max world
        # size for elasticity to match parity with Uru and other systems,
        # although DPP will solve this by dynamically re-sharding.
        return self.num_samples_this_phase

    def _recreate_data_loader_from_dataset(self):
        """
        This utility is invoked to re-create the data loader object
        for the current phase of execution, using the existing dataset.
        This is sufficient when advancing phases.
        """
        logging.info("Recreating data loader for new phase")
        dataset = self.dataloaders[self.phase_type].dataset
        num_workers = 0
        if hasattr(self.dataloaders[self.phase_type], "num_workers"):
            num_workers = self.dataloaders[self.phase_type].num_workers
        pin_memory = False
        if hasattr(self.dataloaders[self.phase_type], "pin_memory"):
            pin_memory = self.dataloaders[self.phase_type].pin_memory
        if self.phase_type == "test":
            current_phase_id = 0
        else:
            current_phase_id = max(self.train_phase_idx, 0)

        self.dataloaders[self.phase_type] = dataset.iterator(
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
        if self.base_model._config.get("freeze_trunk", False):
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
        # TODO (mannatsingh): Figure out how to set the state of the dataloaders
        # Re-build dataloader & re-create iterator.
        self._recreate_data_loader_from_dataset()
        self._reshuffle_data()
        self.create_data_iterator()
        # Set up pytorch module in train vs eval mode, update optimizer.
        self._set_model_train_mode()
