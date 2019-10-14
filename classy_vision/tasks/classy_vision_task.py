#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

from classy_vision.criterions import ClassyCriterion, build_criterion
from classy_vision.dataset import ClassyDataset, build_dataset
from classy_vision.generic.util import update_classy_state
from classy_vision.hooks import ClassyHook
from classy_vision.meters import build_meters
from classy_vision.models import ClassyVisionModel, build_model
from classy_vision.optim import ClassyOptimizer, build_optimizer
from classy_vision.state.classy_state import ClassyState


class ClassyVisionTask(object):
    def __init__(self, num_phases: int):
        self.criterion = None
        self.datasets = {}
        self.meters = []
        self.num_phases = num_phases
        self.test_only = False
        self.reset_heads = False
        self.model = None
        self.optimizer = None
        self.checkpoint = None
        self.phases = []
        self.hooks = []

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

    def set_hooks(self, hooks: List[ClassyHook]):
        self.hooks = hooks
        return self

    def set_model(self, model: ClassyVisionModel):
        self.model = model
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
        optimizer = build_optimizer(optimizer_config, model)

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

        task.reset_heads = config.get("reset_heads", False)
        return task

    def get_config(self):
        return {
            "criterion": self.criterion._config_DO_NOT_USE,
            "dataset": {
                split: dataset._config_DO_NOT_USE
                for split, dataset in self.datasets.items()
            },
            "meters": [meter._config_DO_NOT_USE for meter in self.meters],
            "model": self.model._config_DO_NOT_USE,
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

    def _update_classy_state(self, state, reset_heads, classy_state_dict=None):
        """
        Updates classy state with the provided state dict from a checkpoint.
        """
        if classy_state_dict is not None:
            state_load_success = update_classy_state(
                state, classy_state_dict, reset_heads=reset_heads
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
        self.phases = self._build_phases()
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
        ).set_hooks(self.hooks)

        classy_state_dict = (
            None
            if self.checkpoint is None
            else self.checkpoint.get("classy_state_dict")
        )
        return self._update_classy_state(
            self.state, self.reset_heads, classy_state_dict
        )
