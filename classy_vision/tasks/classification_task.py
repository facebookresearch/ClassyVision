#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import enum
import logging
import time
from typing import Any, Dict, List, Optional, Union

import torch
from classy_vision.dataset import ClassyDataset, build_dataset
from classy_vision.generic.distributed_util import (
    all_reduce_mean,
    barrier,
    init_distributed_data_parallel_model,
    is_distributed_training_run,
)
from classy_vision.generic.util import (
    copy_model_to_gpu,
    recursive_copy_to_gpu,
    update_classy_state,
)
from classy_vision.hooks import ClassyHookFunctions
from classy_vision.losses import ClassyLoss, build_loss
from classy_vision.meters import build_meters
from classy_vision.models import ClassyModel, build_model
from classy_vision.optim import ClassyOptimizer, build_optimizer
from torch.distributed import broadcast

from . import register_task
from .classy_task import ClassyTask


try:
    import apex

    apex_available = True
except ImportError:
    apex_available = False


class BroadcastBuffersMode(enum.Enum):
    DISABLED = enum.auto()
    # Enable DistributedDataParallel's broadcast_buffers option, synchronizing
    # model buffers every forward pass.
    FORWARD_PASS = enum.auto()
    # Similar to FORWARD_PASS, but only synchronizes model buffers once
    # per epoch, between train and test phases. If your motivation for
    # synchronizing buffers is for buffers to be consistent during eval, use
    # this instead of FORWARD_PASS to reduce training overhead.
    BEFORE_EVAL = enum.auto()


@register_task("classification_task")
class ClassificationTask(ClassyTask):
    """Basic classification training task.

    This task encapsultates all of the components and steps needed to
    train a classifier using a :class:`classy_vision.trainer.ClassyTrainer`.

    Assumes a train / test phase per each epoch and that the datasets
    have the same API as the map-style Dataset class in
    `torch.utils.data.dataset <https://pytorch.org/docs/stable/data.html
    #torch.utils.data.Dataset>`_ (in particular, this task makes use of
    the len).  If you are using an `IterableDataset <https://pytorch.org/docs/
    stable/data.html#torch.utils.data.IterableDataset>`_ then a custom task
    may be appropriate.


    :var loss: Loss (see :class:`classy_vision.losses.ClassyLoss`) function used
        for computing the loss in each forward pass
    :var datasets: Mapping from a ``phase_type`` in ["train", "test']
        to dataset used for training (or testing)
    :var meters: List of meters (see :class:`classy_vision.meters.ClassyMeter`)
        to calculate during training
    :var num_epochs: Number of epochs (passes over dataset) to train
    :var test_only: Used to only run the test phase
    :var base_model: Model to be trained, unwrapped in DDP or DP wrappers
    :var optimizer: Optimizer used in train step
    :var checkpoint: Serializable dict which represents state in training
    :var phases: List of phase specific information, e.g. if phase is
        train / test.
    :var hooks: List of hooks to apply during training
    :var train: Phase type, if true it means we are training,
        false means testing
    :var distributed_model: Base model, but wrapped in DDP (DistributedDataParallel)
    :var phase_idx: Current phase id, first phase is 0, if task has not started
        training then returns -1
    :var train_phase_idx: Only counts train phases
    :var num_updates: Number of total parameter updates applied to model
        by the optimizer
    :var data_iterator: Iterator which can be used to obtain batches
    :var losses: Loss curve
    :var perf_log: list of training speed measurements, to be logged

    """

    def __init__(self):
        """Constructs a ClassificationTask
        """
        super().__init__()

        self.loss = None
        self.datasets = {}
        self.meters = []
        self.num_epochs = 1
        self.test_phase_period = 1
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
        self.losses = []
        self.broadcast_buffers_mode: BroadcastBuffersMode = (
            BroadcastBuffersMode.DISABLED
        )
        self.amp_opt_level = None
        self.perf_log = []

    def set_checkpoint(self, checkpoint):
        """Sets checkpoint on task.

        Args:
            checkpoint: A serializable dict representing current task state
        """
        assert (
            checkpoint is None or "classy_state_dict" in checkpoint
        ), "Checkpoint does not contain classy_state_dict"
        self.checkpoint = checkpoint

    def set_num_epochs(self, num_epochs: Union[int, float]):
        """Set number of epochs to be run.

        Args:
           num_epochs: Number of epochs to run task
        """
        self.num_epochs = num_epochs
        return self

    def set_test_phase_period(self, test_phase_period: int):
        """Set the period of test phase.

        Args:
            test_phase_period: The period of test phase
        """
        self.test_phase_period = test_phase_period
        return self

    def set_dataset(self, dataset: ClassyDataset, phase_type: str):
        """Set dataset for phase type on task

        Args:
            dataset: ClassyDataset for returning samples.
            phase_type: str must be one of "train" or "test"
        """
        assert phase_type in [
            "train",
            "test",
        ], "phase_type must be in ['train', 'test']"
        self.datasets[phase_type] = dataset
        return self

    def set_optimizer(self, optimizer: ClassyOptimizer):
        """Set optimizer for task

        Args:
            optimizer: optimizer for task
        """
        self.optimizer = optimizer
        return self

    def set_loss(self, loss: ClassyLoss):
        """Set loss function for task

        Args:
            loss: loss for task
        """
        self.loss = loss
        return self

    def set_meters(self, meters: List["ClassyMeter"]):
        """Set meters for task

        Args:
            meters: list of meters to compute during training
        """
        self.meters = meters
        return self

    def set_distributed_options(self, broadcast_buffers_mode: BroadcastBuffersMode):
        """Set distributed options.

        Args:
            broadcast_buffers_mode: Broadcast buffers mode. See
                :class:`BroadcastBuffersMode` for options.
        """
        self.broadcast_buffers_mode = broadcast_buffers_mode
        return self

    def set_hooks(self, hooks: List["ClassyHook"]):
        """Set hooks for task

        Args:
            hooks: List of hooks to apply during training
        """
        from classy_vision.hooks import ClassyHook

        assert isinstance(hooks, list)
        assert all(isinstance(hook, ClassyHook) for hook in hooks)
        assert len({hook.name() for hook in hooks}) == len(
            hooks
        ), "Cannot have repeated hooks of the same class"

        self.hooks = hooks
        return self

    def set_model(self, model: ClassyModel):
        """Set model for task

        Args:
            model: Model to be trained
        """
        self.base_model = model
        return self

    def set_test_only(self, test_only: bool):
        """Set test only flag

        Args:
            test_only: If true, only test phases will be run
        """
        self.test_only = test_only
        return self

    def set_amp_opt_level(self, opt_level: Optional[str]):
        """Disable / enable apex.amp and set the automatic mixed precision level.

        apex.amp can be utilized for mixed / half precision training.

        Args:
            opt_level: Opt level used to initialize apex.amp. Set to None to disable
                amp. Supported modes are -
                    O0: FP32 training
                    O1: Mixed Precision
                    O2: "Almost FP16" Mixed Precision
                    O3: FP16 training
                See https://nvidia.github.io/apex/amp.html#opt-levels for more info.
        Raises:
            RuntimeError: If opt_level is not None and apex is not installed.

        Warning: apex needs to be installed to utilize this feature.
        """
        if opt_level not in [None, "O0", "O1", "O2", "O3"]:
            raise ValueError(f"Unsupported opt_level: {opt_level}")

        self.amp_opt_level = opt_level

        if opt_level is None:
            logging.info(f"AMP disabled")
        else:
            if not apex_available:
                raise RuntimeError("apex is not installed, cannot enable amp")

            logging.info(f"AMP enabled with opt_level {opt_level}")
        return self

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassificationTask":
        """Instantiates a ClassificationTask from a configuration.

        Args:
            config: A configuration for a ClassificationTask.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A ClassificationTask instance.
        """
        optimizer_config = config["optimizer"]
        optimizer_config["num_epochs"] = config["num_epochs"]

        datasets = {}
        phase_types = ["train", "test"]
        for phase_type in phase_types:
            datasets[phase_type] = build_dataset(config["dataset"][phase_type])
        loss = build_loss(config["loss"])
        test_only = config.get("test_only", False)
        amp_opt_level = config.get("amp_opt_level")
        meters = build_meters(config.get("meters", {}))
        model = build_model(config["model"])
        # put model in eval mode in case any hooks modify model states, it'll
        # be reset to train mode before training
        model.eval()
        optimizer = build_optimizer(optimizer_config)

        task = (
            cls()
            .set_num_epochs(config["num_epochs"])
            .set_test_phase_period(config.get("test_phase_period", 1))
            .set_loss(loss)
            .set_test_only(test_only)
            .set_model(model)
            .set_optimizer(optimizer)
            .set_meters(meters)
            .set_amp_opt_level(amp_opt_level)
            .set_distributed_options(
                BroadcastBuffersMode[config.get("broadcast_buffers", "DISABLED")]
            )
        )
        for phase_type in phase_types:
            task.set_dataset(datasets[phase_type], phase_type)

        return task

    @property
    def num_batches_per_phase(self):
        """Returns number of batches in current phase iterator
        """
        return len(self.data_iterator)

    @property
    def model(self):
        """Returns model used in training (can be wrapped with DDP)
        """
        return (
            self.distributed_model if is_distributed_training_run() else self.base_model
        )

    @property
    def phase_type(self):
        """Returns current phase type. String with value "train" or "test"
        """
        return "train" if self.train else "test"

    @property
    def eval_phase_idx(self):
        """Returns current evaluation phase
        """
        return self.phase_idx - self.train_phase_idx - 1

    def get_data_iterator(self):
        """Returns data iterator for current phase
        """
        return self.data_iterator

    def get_total_training_phases(self):
        """
        Returns the total number of "train" phases in the task
        """
        num_training_phases = 0
        for phase in self.phases:
            if phase["train"] is True:
                num_training_phases += 1
        return num_training_phases

    def get_total_test_phases(self):
        """
        Returns the total number of "test" phases in the task
        """
        num_test_phases = 0
        for phase in self.phases:
            if phase["train"] is False:
                num_test_phases += 1
        return num_test_phases

    def _build_phases(self):
        """Returns list of phases from config.

        These phases will look like:
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
            for i, phase in enumerate(phases):
                final_phases.append(phase)
                if (i + 1) % self.test_phase_period == 0:
                    final_phases.append({"train": False})
            if final_phases[-1]["train"]:
                final_phases.append({"train": False})
            return final_phases

        return [{"train": False} for _ in range(self.num_epochs)]

    def build_dataloader(
        self,
        phase_type,
        num_workers,
        pin_memory,
        multiprocessing_context=None,
        **kwargs,
    ):
        """Buildss a dataloader iterable for a particular phase type.

        Args:
            phase_type: "train" or "test" iterable
            num_workers: Number of dataloading processes. If 0,
                dataloading is done on main process. See `PyTorch dataloader
                documentation <https://pytorch.org/docs/stable/
                data.html#torch.utils.data.DataLoader>`_ for more details on
                ``num_workers`` and the usage
                of python multiprocessing in dataloaders
            pin_memory: if true pin memory on GPU. See PyTorch dataloader
                documentation for details on ``pin_memory``.
            multiprocessing_context: Determines how processes are spawned.
                Value must be one of None, "spawn", "fork", "forkserver".
                If None, then context is inherited from parent process

        Returns:
            Returns a iterable over the dataset
        """
        return self.datasets[phase_type].iterator(
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context,
            **kwargs,
        )

    def build_dataloaders(
        self, num_workers, pin_memory, multiprocessing_context=None, **kwargs
    ):
        """Build a dataloader for each phase type

        Args:
            num_workers: Number of dataloading processes. If 0,
                dataloading is done on main process. See `PyTorch dataloader
                documentation <https://pytorch.org/docs/stable/
                data.html#torch.utils.data.DataLoader>`_
                for more details on num_workers and the usage
                of python multiprocessing in dataloaders
            pin_memory: if true pin memory on GPU. See PyTorch dataloader
                documentation for details on pin_memory.
            multiprocessing_context: Determines how processes are spawned.
                Value must be one of None, "spawn", "fork", "forkserver".
                If None, then context is inherited from parent process

        Returns:
            Returns an iterable over the dataset associated with each phase_type
        """
        return {
            phase_type: self.build_dataloader(
                phase_type,
                num_workers=num_workers,
                pin_memory=pin_memory,
                multiprocessing_context=multiprocessing_context,
                **kwargs,
            )
            for phase_type in self.datasets.keys()
        }

    def prepare(
        self,
        num_dataloader_workers=0,
        pin_memory=False,
        use_gpu=False,
        dataloader_mp_context=None,
    ):
        """Prepares task for training, populates all derived attributes

        Args:
            num_dataloader_workers: Number of dataloading processes. If 0,
                dataloading is done on main process
            pin_memory: if true pin memory on GPU
            use_gpu: if true, load model, optimizer, loss, etc on GPU
            dataloader_mp_context: Determines how processes are spawned.
                Value must be one of None, "spawn", "fork", "forkserver".
                If None, then context is inherited from parent process
        """
        self.phases = self._build_phases()
        self.dataloaders = self.build_dataloaders(
            num_workers=num_dataloader_workers,
            pin_memory=pin_memory,
            multiprocessing_context=dataloader_mp_context,
        )

        # move the model and loss to the right device
        if use_gpu:
            self.base_model, self.loss = copy_model_to_gpu(self.base_model, self.loss)
        else:
            self.loss.cpu()
            self.base_model.cpu()

        # initialize the pytorch optimizer now since the model has been moved to
        # the appropriate device
        self.optimizer.init_pytorch_optimizer(self.base_model, loss=self.loss)

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

        if self.amp_opt_level is not None:
            # Initialize apex.amp. This updates the model and the PyTorch optimizer (
            # which is wrapped by the ClassyOptimizer in self.optimizer)
            self.base_model, self.optimizer.optimizer = apex.amp.initialize(
                self.base_model, self.optimizer.optimizer, opt_level=self.amp_opt_level
            )
        self.init_distributed_data_parallel_model()

    def init_distributed_data_parallel_model(self):
        """
        Initialize
        `torch.nn.parallel.distributed.DistributedDataParallel <https://pytorch.org/
        docs/stable/nn.html#distributeddataparallel>`_.

        Needed for distributed training. This is where a model should be wrapped by DDP.
        """
        if not is_distributed_training_run():
            return
        assert (
            self.distributed_model is None
        ), "init_ddp_non_elastic must only be called once"

        broadcast_buffers = (
            self.broadcast_buffers_mode == BroadcastBuffersMode.FORWARD_PASS
        )
        self.distributed_model = init_distributed_data_parallel_model(
            self.base_model, broadcast_buffers=broadcast_buffers
        )
        if isinstance(self.loss, ClassyLoss) and self.loss.has_learned_parameters():
            logging.info("Initializing distributed loss")
            self.loss = init_distributed_data_parallel_model(
                self.loss, broadcast_buffers=broadcast_buffers
            )

    @property
    def where(self):
        """Returns the proportion of training that has completed. If in test
        only mode, returns proportion of testing completed

        Returned value is a float in the range [0, 1)
        """
        current_step = self.num_updates / self.get_global_batchsize()
        num_phases = (
            self.get_total_test_phases()
            if self.test_only
            else self.get_total_training_phases()
        )

        if self.num_batches_per_phase <= 0:
            raise RuntimeError("No batches to read. Is the dataset empty?")

        num_steps = num_phases * self.num_batches_per_phase
        where = current_step / num_steps

        assert where >= 0 and where < 1, f"Invalid where: {where}"

        return where

    def get_classy_state(self, deep_copy: bool = False):
        """Returns serialiable state of task

        Args:
            deep_copy: If true, does a deep copy of state before returning.
        """
        classy_state_dict = {
            "train": self.train,
            "base_model": self.base_model.get_classy_state(),
            "meters": [meter.get_classy_state() for meter in self.meters],
            "optimizer": self.optimizer.get_classy_state(),
            "phase_idx": self.phase_idx,
            "train_phase_idx": self.train_phase_idx,
            "num_updates": self.num_updates,
            "losses": self.losses,
            "hooks": {hook.name(): hook.get_classy_state() for hook in self.hooks},
            "loss": {},
        }
        if isinstance(self.loss, ClassyLoss):
            classy_state_dict["loss"] = self.loss.get_classy_state()
        if deep_copy:
            classy_state_dict = copy.deepcopy(classy_state_dict)
        return classy_state_dict

    def set_classy_state(self, state):
        """Set task state

        Args:
            state: Dict containing state of a task
        """
        # some settings are different in test only
        self.train = False if self.test_only else state["train"]
        if not self.test_only:
            self.phase_idx = state["phase_idx"]
            self.num_updates = state["num_updates"]
            self.train_phase_idx = state["train_phase_idx"]
            self.losses = state["losses"]
            for meter, meter_state in zip(self.meters, state["meters"]):
                meter.set_classy_state(meter_state)

        self.base_model.set_classy_state(state["base_model"])
        self.optimizer.set_classy_state(state["optimizer"])
        if state.get("loss") and isinstance(self.loss, ClassyLoss):
            self.loss.set_classy_state(state["loss"])

        for hook in self.hooks:
            # we still want to be able to run when new hooks are added or old
            # hooks are removed
            if hook.name() in state["hooks"]:
                hook.set_classy_state(state["hooks"][hook.name()])
            else:
                logging.warn(f"No state found for hook: {hook.name()}")
        # TODO (mannatsingh): Figure out how to set the state of the dataloaders
        # Re-build dataloader & re-create iterator.
        self._recreate_data_loader_from_dataset()
        self.create_data_iterator()
        # Set up pytorch module in train vs eval mode, update optimizer.
        self._set_model_train_mode()

    def eval_step(self, use_gpu, local_variables=None):
        if local_variables is None:
            local_variables = {}

        # Process next sample
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

        # Copy sample to GPU
        local_variables["target"] = local_variables["sample"]["target"]
        if use_gpu:
            for key, value in local_variables["sample"].items():
                local_variables["sample"][key] = recursive_copy_to_gpu(
                    value, non_blocking=True
                )

        with torch.no_grad():
            local_variables["output"] = self.model(local_variables["sample"]["input"])

            local_variables["local_loss"] = self.compute_loss(
                local_variables["output"], local_variables["sample"]
            )

            local_variables["loss"] = local_variables["local_loss"].detach().clone()
            local_variables["loss"] = all_reduce_mean(local_variables["loss"])

            self.losses.append(
                local_variables["loss"].data.cpu().item()
                * local_variables["target"].size(0)
            )

            self.update_meters(local_variables["output"], local_variables["sample"])

    def train_step(self, use_gpu, local_variables=None):
        """Train step to be executed in train loop

        Args:
            use_gpu: if true, execute training on GPU
            local_variables: Dict containing intermediate values
                in train_step for access by hooks
        """

        if local_variables is None:
            local_variables = {}

        # Process next sample
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

        # Copy sample to GPU
        local_variables["target"] = local_variables["sample"]["target"]
        if use_gpu:
            for key, value in local_variables["sample"].items():
                local_variables["sample"][key] = recursive_copy_to_gpu(
                    value, non_blocking=True
                )

        with torch.enable_grad():
            # Forward pass
            local_variables["output"] = self.model(local_variables["sample"]["input"])

            local_variables["local_loss"] = self.compute_loss(
                local_variables["output"], local_variables["sample"]
            )

            local_variables["loss"] = local_variables["local_loss"].detach().clone()
            local_variables["loss"] = all_reduce_mean(local_variables["loss"])

            self.losses.append(
                local_variables["loss"].data.cpu().item()
                * local_variables["target"].size(0)
            )

            self.update_meters(local_variables["output"], local_variables["sample"])

        # Run backwards pass / update optimizer
        if self.amp_opt_level is not None:
            self.optimizer.zero_grad()
            with apex.amp.scale_loss(
                local_variables["local_loss"], self.optimizer.optimizer
            ) as scaled_loss:
                scaled_loss.backward()
        else:
            self.optimizer.backward(local_variables["local_loss"])

        self.optimizer.update_schedule_on_step(self.where)
        self.optimizer.step()

        self.num_updates += self.get_global_batchsize()

    def compute_loss(self, model_output, sample):
        return self.loss(model_output, sample["target"])

    def update_meters(self, model_output, sample):
        target = sample["target"].detach().cpu()
        model_output = model_output.detach().cpu()

        # Update meters
        for meter in self.meters:
            meter.update(model_output, target, is_train=self.train)

    def advance_phase(self):
        """Performs bookkeeping / task updates between phases

        Increments phase idx, resets meters, resets loss history,
        resets counters, shuffles dataset, rebuilds iterators, and
        sets the train / test state for phase.
        """
        logging.info("Advancing phase")
        # Reset meters for next phase / epoch
        for meter in self.meters:
            meter.reset()

        # Reset loss history for next epoch
        self.losses = []

        # Setup new phase
        self.phase_idx += 1
        phase = self.phases[self.phase_idx]
        self.train = True if phase["train"] else False
        if self.train:
            self.train_phase_idx += 1

        # Re-build dataloader & re-create iterator anytime membership changes.
        self._recreate_data_loader_from_dataset()
        self.create_data_iterator()
        # Set up pytorch module in train vs eval mode, update optimizer.
        self._set_model_train_mode()

        # Update the optimizer schedule
        if self.train and self.train_phase_idx >= 0:
            self.optimizer.update_schedule_on_epoch(self.where)

    def done_training(self):
        """Stop condition for training
        """
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
        multiprocessing_context = None
        if hasattr(self.dataloaders[phase_type], "multiprocessing_context"):
            multiprocessing_context = self.dataloaders[
                phase_type
            ].multiprocessing_context
        if phase_type == "test":
            current_phase_id = 0
        else:
            current_phase_id = max(self.train_phase_idx, 0)

        self.dataloaders[phase_type] = self.build_dataloader(
            phase_type=phase_type,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context,
            current_phase_id=current_phase_id,
        )

    def create_data_iterator(self):
        """Creates data iterator for phase.
        """
        # Delete iterator explicitly so that all dataloader processes
        # are cleaned up.
        del self.data_iterator
        self.data_iterator = iter(self.dataloaders[self.phase_type])

    def _set_model_train_mode(self):
        """Set train mode for model
        """
        phase = self.phases[self.phase_idx]
        self.base_model.train(phase["train"])
        self.loss.train(phase["train"])

        if (
            self.broadcast_buffers_mode == BroadcastBuffersMode.BEFORE_EVAL
            and not self.train
        ):
            self._broadcast_buffers()

    def _broadcast_buffers(self):
        """Explicitly synchronize buffers across all devices."""
        if self.distributed_model is None:
            return
        buffers = list(self.base_model.buffers())
        if len(buffers) > 0:
            logging.info("Synchronizing buffers before evaluation.")
            for buffer in buffers:
                broadcast(buffer, 0, group=self.distributed_model.process_group)

    # TODO: Functions below should be better abstracted into the dataloader
    # abstraction
    def get_batchsize_per_replica(self):
        """Return local replica's batchsize for dataset (e.g. batchsize per GPU)
        """
        # TODO(T47573564) - cleaner abstraction
        return self.dataloaders[self.phase_type].dataset.get_batchsize_per_replica()

    def get_global_batchsize(self):
        """Return global batchsize across all trainers
        """
        return self.dataloaders[self.phase_type].dataset.get_global_batchsize()

    def on_start(self, local_variables):
        self.run_hooks(local_variables, ClassyHookFunctions.on_start.name)

    def on_phase_start(self, local_variables):
        self.phase_start_time_total = time.perf_counter()

        self.advance_phase()

        self.run_hooks(local_variables, ClassyHookFunctions.on_phase_start.name)

        self.phase_start_time_train = time.perf_counter()

    def on_phase_end(self, local_variables):
        self.log_phase_end("train")

        logging.info("Syncing meters on phase end...")
        for meter in self.meters:
            meter.sync_state()
        logging.info("...meters synced")
        barrier()

        self.run_hooks(local_variables, ClassyHookFunctions.on_phase_end.name)
        self.perf_log = []

        self.log_phase_end("total")

    def on_end(self, local_variables):
        self.run_hooks(local_variables, ClassyHookFunctions.on_end.name)

    def log_phase_end(self, tag):
        if not self.train:
            return

        start_time = (
            self.phase_start_time_train
            if tag == "train"
            else self.phase_start_time_total
        )
        phase_duration = time.perf_counter() - start_time
        im_per_sec = (
            self.get_global_batchsize() * self.num_batches_per_phase
        ) / phase_duration
        self.perf_log.append(
            {
                "tag": tag,
                "phase_idx": self.train_phase_idx,
                "epoch_duration": phase_duration,
                "im_per_sec": im_per_sec,
            }
        )
