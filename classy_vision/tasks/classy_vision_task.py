#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

import torch
from classy_vision.criterions import ClassyCriterion, build_criterion
from classy_vision.dataset import build_dataset
from classy_vision.generic.util import copy_model_to_gpu, update_classy_state
from classy_vision.meters import build_meter
from classy_vision.models import build_model
from classy_vision.optim import build_optimizer
from classy_vision.state.classy_state import ClassyState


class ClassyVisionTask(object):
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        device_type: str,
        meter_config: Dict[str, Any],
        model_config: Dict[str, Any],
        num_phases: int,
        num_workers: int,
        optimizer_config: Dict[str, Any],
        pin_memory: bool,
        test_only: bool,
    ):
        self.criterion = None
        self.dataset_config = dataset_config
        self.device_type = device_type
        self.meter_config = meter_config
        self.model_config = model_config
        self.num_phases = num_phases
        self.num_workers = num_workers
        self.optimizer_config = optimizer_config
        self.pin_memory = pin_memory
        self.test_only = test_only

        self.checkpoint = None
        self.dataloaders = self.build_dataloaders()
        self.phases = self._build_phases()
        self.model = self._build_model()
        self.optimizer = build_optimizer(self.optimizer_config, self.model)
        self.meters = self._build_meters()

    def set_criterion(self, criterion: ClassyCriterion):
        self.criterion = criterion
        return self

    @classmethod
    def setup_task(cls, config, args, **kwargs):
        """
        Setup the task using config. Validate that models / datasets /
        losses / meters are compatible
        """

        # allow some command-line options to override configuration:
        if "machine" not in config:
            config["machine"] = {}
        if "device" not in config["machine"]:
            config["machine"]["device"] = args.device
        if "num_workers" not in config["machine"]:
            config["machine"]["num_workers"] = args.num_workers
        if "test_only" not in config:
            config["test_only"] = args.test_only
        config["machine"]["pin_memory"] = (
            config["machine"]["device"] == "gpu" and torch.cuda.device_count() > 1
        )

        return cls.from_config(config)

    @classmethod
    def from_config(cls, config):
        optimizer_config = config["optimizer"]
        optimizer_config["num_epochs"] = config["num_phases"]

        criterion = build_criterion(config["criterion"])
        return cls(
            dataset_config=config["dataset"],
            device_type=config["machine"]["device"],
            meter_config=config.get("meters", {}),
            model_config=config["model"],
            num_phases=config["num_phases"],
            num_workers=config["machine"]["num_workers"],
            optimizer_config=optimizer_config,
            pin_memory=config["machine"]["pin_memory"],
            test_only=config["test_only"],
        ).set_criterion(criterion)

    def get_config(self):
        return {
            "criterion": self.criterion._config_DO_NOT_USE,
            "dataset": self.dataset_config,
            "device_type": self.device_type,
            "meters": self.meter_config,
            "model": self.model_config,
            "num_phases": self.num_phases,
            "num_workers": self.num_workers,
            "optimizer": self.optimizer_config,
            "pin_memory": self.pin_memory,
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

    def build_dataloaders(self, **kwargs):
        """
        Returns dataloader (with splits such as train / test) for task.
        """
        # Add machine params to dataloader config:
        return {
            split: build_dataset(self.dataset_config[split]).iterator(
                **kwargs, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            for split in ["train", "test"]
        }

    def _build_model(self):
        """
        Returns model for task.
        """
        # TODO (aadcock): Need to make models accept target metadata
        # as build param to support non-classification tasks
        model = build_model(self.model_config)

        if self.device_type == "gpu":
            model = copy_model_to_gpu(model)

        # put model in eval mode in case any hooks modify model states, it' will
        # be reset to train mode before training
        model.eval()

        return model

    def _build_meters(self):
        """
        Returns meters for task.
        """
        configs = [{"name": name, **args} for name, args in self.meter_config.items()]
        return [build_meter(config) for config in configs]

    def _update_classy_state(self, state, classy_state_dict=None):
        """
        Updates classy state with the provided state dict from a checkpoint.
        """
        if classy_state_dict is not None:
            state_load_success = update_classy_state(
                state,
                classy_state_dict,
                is_finetuning=self.model_config.get("is_finetuning", False),
                test_only=self.test_only,
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

    def build_initial_state(self):
        """
        Creates initial state using config.
        """
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
