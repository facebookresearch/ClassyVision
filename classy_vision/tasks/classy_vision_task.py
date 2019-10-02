#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

import torch
from classy_vision.criterions import ClassyCriterion, build_criterion
from classy_vision.dataset import build_dataset
from classy_vision.generic.util import update_classy_state
from classy_vision.meters import build_meters
from classy_vision.models import build_model
from classy_vision.optim import build_optimizer
from classy_vision.state.classy_state import ClassyState


class ClassyVisionTask(object):
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        model_config: Dict[str, Any],
        num_phases: int,
        optimizer_config: Dict[str, Any],
    ):
        self.criterion = None
        self.dataset_config = dataset_config
        self.meters = []
        self.model_config = model_config
        self.num_phases = num_phases
        self.optimizer_config = optimizer_config
        self.test_only = False

        self.checkpoint = None
        self.datasets = self.build_datasets()
        self.phases = self._build_phases()
        self.model = self._build_model()
        self.optimizer = build_optimizer(self.optimizer_config, self.model)

    def set_criterion(self, criterion: ClassyCriterion):
        self.criterion = criterion
        return self

    def set_test_only(self, test_only: bool):
        self.test_only = test_only
        return self

    def set_meters(self, meters: List["ClassyMeter"]):
        self.meters = meters
        return self

    @classmethod
    def from_config(cls, config):
        optimizer_config = config["optimizer"]
        optimizer_config["num_epochs"] = config["num_phases"]

        criterion = build_criterion(config["criterion"])
        test_only = config["test_only"]
        meters = build_meters(config.get("meters", {}))
        return (
            cls(
                dataset_config=config["dataset"],
                model_config=config["model"],
                num_phases=config["num_phases"],
                optimizer_config=optimizer_config,
            )
            .set_criterion(criterion)
            .set_test_only(test_only)
            .set_meters(meters)
        )

    def get_config(self):
        return {
            "criterion": self.criterion._config_DO_NOT_USE,
            "dataset": self.dataset_config,
            "meters": [meter._config_DO_NOT_USE for meter in self.meters],
            "model": self.model_config,
            "num_phases": self.num_phases,
            "optimizer": self.optimizer_config,
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

    def build_datasets(self):
        return {
            split: build_dataset(self.dataset_config[split])
            for split in ["train", "test"]
        }

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

    def _build_model(self):
        """
        Returns model for task.
        """
        # TODO (aadcock): Need to make models accept target metadata
        # as build param to support non-classification tasks
        model = build_model(self.model_config)

        # put model in eval mode in case any hooks modify model states, it' will
        # be reset to train mode before training
        model.eval()

        return model

    def _update_classy_state(self, state, classy_state_dict=None):
        """
        Updates classy state with the provided state dict from a checkpoint.
        """
        if classy_state_dict is not None:
            state_load_success = update_classy_state(
                state,
                classy_state_dict,
                reset_heads=self.model_config.get("reset_heads", True),
            )
            assert (
                state_load_success
            ), "Update classy state from checkpoint was unsuccessful."
        return state

    def set_checkpoint(self, checkpoint):
        assert (
            checkpoint is None or "classy_state_dict" in checkpoint
        ), "Checkpoint does not contain classy_state_dict"
        self.checkpoint = checkpoint

    def build_initial_state(self, num_workers=0, pin_memory=False):
        """
        Creates initial state using config.
        """
        self.dataloaders = self.build_dataloaders(
            num_workers=num_workers, pin_memory=pin_memory
        )
        self.state = ClassyState(
            self,
            self.phases,
            self.phases[0]["train"],
            self.dataloaders,
            self.model,
            self.criterion,
            self.meters,
            self.optimizer,
        )

        classy_state_dict = (
            None
            if self.checkpoint is None
            else self.checkpoint.get("classy_state_dict")
        )
        return self._update_classy_state(self.state, classy_state_dict)
