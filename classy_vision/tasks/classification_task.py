#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import enum
import json
import logging
import math
import multiprocessing as mp
import time
from typing import Any, Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn as nn
from classy_vision.dataset import ClassyDataset, build_dataset
from classy_vision.dataset.transforms.mixup import MixupTransform
from classy_vision.generic.distributed_util import (
    all_reduce_mean,
    barrier,
    init_distributed_data_parallel_model,
    is_distributed_training_run,
)
from classy_vision.generic.util import (
    Timer,
    copy_model_to_gpu,
    load_and_broadcast_checkpoint,
    master_params,
    recursive_copy_to_gpu,
    split_batchnorm_params,
    update_classy_state,
)
from classy_vision.hooks import CheckpointHook, ClassyHook, build_hooks
from classy_vision.losses import ClassyLoss, build_loss
from classy_vision.meters import ClassyMeter, build_meters
from classy_vision.models import ClassyModel, build_model
from classy_vision.optim import (
    ClassyOptimizer,
    build_optimizer,
    build_optimizer_schedulers,
)
from torch.distributed import broadcast

from . import register_task
from .classy_task import ClassyTask


try:
    import apex

    apex_available = True
except ImportError:
    apex_available = False

try:
    from torch.cuda.amp import GradScaler as TorchGradScaler

except ImportError:
    pass


class AmpType(enum.Enum):
    # Automatic Mixed Precision supported types
    APEX = enum.auto()
    PYTORCH = enum.auto()


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


class BatchNormSyncMode(enum.Enum):
    DISABLED = enum.auto()  # No Synchronized Batch Normalization
    PYTORCH = enum.auto()  # Use torch.nn.SyncBatchNorm
    APEX = enum.auto()  # Use apex.parallel.SyncBatchNorm, needs apex to be installed


class LastBatchInfo(NamedTuple):
    loss: torch.Tensor
    output: torch.Tensor
    target: torch.Tensor
    sample: Dict[str, Any]
    step_data: Dict[str, Any]


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
    :var optimizer_schedulers: Dictionary. Key is the name of the optimizer
        option (e.g. lr), value is a ClassyParamScheduler
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
    :var clip_grad_norm: maximum gradient norm (default None)
    :var simulated_global_batchsize: batch size simulated via gradient accumulation
    :var optimizer_period: apply optimizer after this many steps; derived from
        simulated_global_batchsize, default 1.
    """

    def __init__(self):
        """Constructs a ClassificationTask"""
        super().__init__()

        self.base_loss = None
        self.datasets = {}
        self.meters = []
        self.num_epochs = 1
        self.test_phase_period = 1
        self.train_phases_per_epoch = 0
        self.test_only = False
        self.base_model = None
        self.optimizer = None
        self.optimizer_schedulers = {}
        self.checkpoint_dict = None
        self.checkpoint_path = None
        self.phases = []
        self.hooks = []
        self.train = True
        self.distributed_model = None
        self.distributed_loss = None
        self.phase_idx = -1
        self.train_phase_idx = -1
        self.num_updates = 0
        self.dataloader = None
        self.data_iterator = None
        self.losses = []
        self.broadcast_buffers_mode: BroadcastBuffersMode = (
            BroadcastBuffersMode.BEFORE_EVAL
        )
        self.amp_args = None
        self.amp_type = None
        self.amp_grad_scaler = None
        self.mixup_transform = None
        self.perf_log = []
        self.last_batch = None
        self.batch_norm_sync_mode = BatchNormSyncMode.DISABLED
        self.find_unused_parameters = True
        self.use_gpu = torch.cuda.is_available()
        self.dataloader_mp_context = "spawn"
        self.bn_weight_decay = False
        self._train_only = True
        self.clip_grad_norm = None
        self.simulated_global_batchsize = None
        self.optimizer_period = 1
        self.ddp_bucket_cap_mb = 25

    def set_use_gpu(self, use_gpu: bool):
        self.use_gpu = use_gpu

        assert (
            not self.use_gpu or torch.cuda.is_available()
        ), "CUDA required to train on GPUs"

        return self

    def set_clip_grad_norm(self, clip_grad_norm: Optional[float]):
        """Sets maximum gradient norm.

        None means gradient clipping is disabled. Defaults to None."""
        self.clip_grad_norm = clip_grad_norm
        if clip_grad_norm is None:
            logging.info("Disabled gradient norm clipping.")
        else:
            logging.info(
                f"Enabled gradient norm clipping with threshold: {clip_grad_norm}"
            )
        return self

    def set_simulated_global_batchsize(self, simulated_global_batchsize: Optional[int]):
        """Sets a simulated batch size by gradient accumulation.

        Gradient accumulation adds up gradients from multiple minibatches and
        steps the optimizer every N train_steps, where N is optimizer_period.
        When enabled, the very last train_steps might end up not updating the
        model, depending on the number of total steps. None means gradient
        accumulation is disabled. Defaults to None."""
        self.simulated_global_batchsize = simulated_global_batchsize
        return self

    def set_checkpoint(self, checkpoint_path: str):
        """Sets checkpoint on task.

        Args:
            checkpoint_path: The path to load the checkpoint from. Can be a file or a
            directory. See :func:`load_checkpoint` for more information.
        """
        self.checkpoint_path = checkpoint_path
        return self

    def _set_checkpoint_dict(self, checkpoint_dict: Dict[str, Any]):
        """Sets the checkpoint dict in the task. Only used for testing.

        Args:
            checkpoint_dict: A serializable dict representing current task state
        """
        self.checkpoint_dict = checkpoint_dict
        return self

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
        if phase_type == "train":
            self.train_phases_per_epoch = getattr(dataset, "phases_per_epoch", 1)
        else:
            self._train_only = False
        return self

    def set_dataloader_mp_context(self, dataloader_mp_context: Optional[str]):
        """Set the multiprocessing context used by the dataloader.

        The context can be either 'spawn', 'fork', 'forkserver' or None (uses the
        default context). See
        https://docs.python.org/3/library/multiprocessing.html#multiprocessing.get_context
        for more details."""

        self.dataloader_mp_context = dataloader_mp_context
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
        self.base_loss = loss
        return self

    def set_meters(self, meters: List["ClassyMeter"]):
        """Set meters for task

        Args:
            meters: list of meters to compute during training
        """
        self.meters = meters
        return self

    def set_distributed_options(
        self,
        broadcast_buffers_mode: BroadcastBuffersMode = BroadcastBuffersMode.BEFORE_EVAL,
        batch_norm_sync_mode: BatchNormSyncMode = BatchNormSyncMode.DISABLED,
        batch_norm_sync_group_size: int = 0,
        find_unused_parameters: bool = True,
        bucket_cap_mb: int = 25,
    ):
        """Set distributed options.

        Args:
            broadcast_buffers_mode: Broadcast buffers mode. See
                :class:`BroadcastBuffersMode` for options.
            batch_norm_sync_mode: Batch normalization synchronization mode. See
                :class:`BatchNormSyncMode` for options.
            batch_norm_sync_group_size: Group size to use for synchronized batch norm.
                0 means that the stats are synchronized across all replicas. For
                efficient synchronization, set it to the number of GPUs in a node (
                usually 8).
            find_unused_parameters: See
                :class:`torch.nn.parallel.DistributedDataParallel` for information.
            bucket_cap_mb: See
                :class:`torch.nn.parallel.DistributedDataParallel` for information.
        Raises:
            RuntimeError: If batch_norm_sync_mode is `BatchNormSyncMode.APEX` and apex
                is not installed.
        """
        self.broadcast_buffers_mode = broadcast_buffers_mode

        if batch_norm_sync_group_size > 0:
            if not batch_norm_sync_mode == BatchNormSyncMode.APEX:
                # this should ideally work with PyTorch Sync BN as well, but it
                # fails while initializing DDP for some reason.
                raise ValueError(
                    "batch_norm_sync_group_size can be > 0 only when "
                    "Apex Synchronized Batch Normalization is being used."
                )
        self.batch_norm_sync_group_size = batch_norm_sync_group_size

        if batch_norm_sync_mode == BatchNormSyncMode.DISABLED:
            logging.info("Synchronized Batch Normalization is disabled")
        else:
            if batch_norm_sync_mode == BatchNormSyncMode.APEX and not apex_available:
                raise RuntimeError("apex is not installed")
            msg = f"Using Synchronized Batch Normalization using {batch_norm_sync_mode}"
            if self.batch_norm_sync_group_size > 0:
                msg += f" and group size {batch_norm_sync_group_size}"
            logging.info(msg)
        self.batch_norm_sync_mode = batch_norm_sync_mode

        self.find_unused_parameters = find_unused_parameters
        self.ddp_bucket_cap_mb = bucket_cap_mb

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
        # TODO (zyan3): we move checkpoint hook to the end of the list because some hooks
        # may change the state of the model, and we want to save changed state in the checkpoint.
        # This is temporary fix.
        non_checkpoint_hooks = [
            hook for hook in hooks if not isinstance(hook, CheckpointHook)
        ]
        checkpoint_hooks = [hook for hook in hooks if isinstance(hook, CheckpointHook)]
        hooks = non_checkpoint_hooks + checkpoint_hooks
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

    def set_bn_weight_decay(self, bn_weight_decay: bool):
        assert type(bn_weight_decay) == bool

        self.bn_weight_decay = bn_weight_decay
        return self

    def set_amp_args(self, amp_args: Optional[Dict[str, Any]]):
        """Disable / enable apex.amp and set the automatic mixed precision parameters.

        apex.amp can be utilized for mixed / half precision training.

        Args:
            amp_args: Dictionary containing arguments to be passed to
            amp.initialize. Set to None to disable amp.  To enable mixed
            precision training, pass amp_args={"opt_level": "O1"} here.
            See https://nvidia.github.io/apex/amp.html for more info.

        Raises:
            RuntimeError: If opt_level is not None and apex is not installed.

        Warning: apex needs to be installed to utilize this feature.
        """
        self.amp_args = amp_args

        if amp_args is None:
            logging.info("AMP disabled")
        else:
            # Check that the requested AMP type is known
            try:
                self.amp_type = AmpType[self.amp_args["amp_type"].upper()]
            except KeyError:
                logging.info("AMP type not specified, defaulting to Apex")
                self.amp_type = AmpType.APEX

            # Check for CUDA availability, required for both Apex and Pytorch AMP
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "AMP is required but CUDA is not supported, cannot enable AMP"
                )

            # Check for Apex availability
            if self.amp_type == AmpType.APEX and not apex_available:
                raise RuntimeError(
                    "Apex AMP is required but Apex is not installed, cannot enable AMP"
                )

            # Set Torch AMP grad scaler, used to prevent gradient underflow
            elif self.amp_type == AmpType.PYTORCH:
                self.amp_grad_scaler = TorchGradScaler()

            logging.info(f"AMP enabled with args {amp_args}")
        return self

    def set_mixup_transform(self, mixup_transform: Optional["MixupTransform"]):
        """Disable / enable mixup transform for data augmentation

        Args::
            mixup_transform: a callable object which performs mixup data augmentation
        """
        self.mixup_transform = mixup_transform
        if mixup_transform is None:
            logging.info("mixup disabled")
        else:
            logging.info("mixup enabled")
        return self

    def set_optimizer_schedulers(self, schedulers):
        self.optimizer_schedulers = schedulers
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
        test_only = config.get("test_only", False)
        if not test_only:
            # TODO Make distinction between epochs and phases in optimizer clear
            train_phases_per_epoch = config["dataset"]["train"].get(
                "phases_per_epoch", 1
            )

            optimizer_config = config["optimizer"]
            optimizer_config["num_epochs"] = (
                config["num_epochs"] * train_phases_per_epoch
            )
            optimizer = build_optimizer(optimizer_config)
            param_schedulers = build_optimizer_schedulers(optimizer_config)

        datasets = {}
        phase_types = ["train", "test"]
        for phase_type in phase_types:
            if phase_type in config["dataset"]:
                datasets[phase_type] = build_dataset(config["dataset"][phase_type])
        loss = build_loss(config["loss"])
        amp_args = config.get("amp_args")
        meters = build_meters(config.get("meters", {}))
        model = build_model(config["model"])

        mixup_transform = None
        if config.get("mixup") is not None:
            assert "alpha" in config["mixup"], "key alpha is missing in mixup dict"
            mixup_transform = MixupTransform(
                config["mixup"]["alpha"], config["mixup"].get("num_classes")
            )

        # hooks config is optional
        hooks_config = config.get("hooks")
        hooks = []
        if hooks_config is not None:
            hooks = build_hooks(hooks_config)

        distributed_config = config.get("distributed", {})
        distributed_options = {
            "broadcast_buffers_mode": BroadcastBuffersMode[
                distributed_config.get("broadcast_buffers", "before_eval").upper()
            ],
            "batch_norm_sync_mode": BatchNormSyncMode[
                distributed_config.get("batch_norm_sync_mode", "disabled").upper()
            ],
            "batch_norm_sync_group_size": distributed_config.get(
                "batch_norm_sync_group_size", 0
            ),
            "find_unused_parameters": distributed_config.get(
                "find_unused_parameters", True
            ),
            "bucket_cap_mb": distributed_config.get("bucket_cap_mb", 25),
        }

        task = (
            cls()
            .set_num_epochs(config["num_epochs"])
            .set_test_phase_period(config.get("test_phase_period", 1))
            .set_loss(loss)
            .set_test_only(test_only)
            .set_model(model)
            .set_meters(meters)
            .set_amp_args(amp_args)
            .set_mixup_transform(mixup_transform)
            .set_distributed_options(**distributed_options)
            .set_hooks(hooks)
            .set_bn_weight_decay(config.get("bn_weight_decay", False))
            .set_clip_grad_norm(config.get("clip_grad_norm"))
            .set_simulated_global_batchsize(config.get("simulated_global_batchsize"))
        )

        if not test_only:
            task.set_optimizer(optimizer)
            task.set_optimizer_schedulers(param_schedulers)

        use_gpu = config.get("use_gpu")
        if use_gpu is not None:
            task.set_use_gpu(use_gpu)

        for phase_type in datasets:
            task.set_dataset(datasets[phase_type], phase_type)

        # NOTE: this is a private member and only meant to be used for
        # logging/debugging purposes. See __repr__ implementation
        task._config = config

        return task

    @property
    def num_batches_per_phase(self):
        """Returns number of batches in current phase iterator"""
        return len(self.data_iterator)

    @property
    def model(self):
        """Returns model used in training (can be wrapped with DDP)"""
        return (
            self.distributed_model if is_distributed_training_run() else self.base_model
        )

    @property
    def loss(self):
        """Returns loss used in training (can be wrapped with DDP)"""
        return self.distributed_loss if self.distributed_loss else self.base_loss

    @property
    def phase_type(self):
        """Returns current phase type. String with value "train" or "test" """
        return "train" if self.train else "test"

    @property
    def eval_phase_idx(self):
        """Returns current evaluation phase"""
        return self.phase_idx - self.train_phase_idx - 1

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

        - If this is a test only run, then only test phases will be
        generated
        - If this is a training run with both train and test datasets, then x phases =
          x train phases + x test phases, interleaved. If test_phase_period > 1, test
          phases are only added after test_phase_period train phases. The last phase is
          always a test phase.
        - If this is a training run with only a train dataset, then x phases = x train
          phases.
        """
        if not self.test_only:
            phases = [
                {"train": True}
                for _ in range(math.ceil(self.train_phases_per_epoch * self.num_epochs))
            ]

            if self._train_only:
                return phases

            final_phases = []
            for i, phase in enumerate(phases):
                final_phases.append(phase)
                if (i + 1) % self.test_phase_period == 0:
                    final_phases.append({"train": False})
            if final_phases[-1]["train"]:
                final_phases.append({"train": False})
            return final_phases

        return [{"train": False} for _ in range(self.num_epochs)]

    def build_dataloader_from_dataset(self, dataset, **kwargs):
        """Builds a dataloader from the provided dataset

        Args:
            dataset: A ClassyDataset
            kwargs: Additional kwargs to pass during dataloader construction for
                derived classes
        """
        return dataset.iterator(
            phase_type=self.phase_type,
            current_phase_id=self.train_phase_idx if self.train else 0,
            pin_memory=self.use_gpu and torch.cuda.device_count() > 1,
            multiprocessing_context=mp.get_context(self.dataloader_mp_context),
            **kwargs,
        )

    def build_dataloaders_for_current_phase(self):
        """Builds dataloader(s) for the current phase.

        Deriving classes can override this method to support custom behavior, like
        supporting multiple dataloaders in parallel.
        """
        self.dataloader = self.build_dataloader_from_dataset(
            self.datasets[self.phase_type]
        )

    def prepare_optimizer(self, optimizer, model, loss=None):
        bn_params, other_params = split_batchnorm_params(model)
        if loss is not None:
            bn_params_loss, params_loss = split_batchnorm_params(loss)
            bn_params = bn_params + bn_params_loss
            other_params = other_params + params_loss

        bn_schedulers = self.optimizer_schedulers.copy()
        if not self.bn_weight_decay:
            bn_schedulers["weight_decay"] = 0

        param_groups = [{"params": other_params, **self.optimizer_schedulers}]
        if len(bn_params) > 0:
            param_groups.append({"params": bn_params, **bn_schedulers})
        self.optimizer.set_param_groups(param_groups)

    def prepare(self):
        """Prepares task for training, populates all derived attributes """

        self.phases = self._build_phases()
        self.train = False if self.test_only else self.train

        if self.batch_norm_sync_mode == BatchNormSyncMode.PYTORCH:
            self.base_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.base_model)
        elif self.batch_norm_sync_mode == BatchNormSyncMode.APEX:
            sync_bn_process_group = apex.parallel.create_syncbn_process_group(
                self.batch_norm_sync_group_size
            )
            self.base_model = apex.parallel.convert_syncbn_model(
                self.base_model, process_group=sync_bn_process_group
            )

        # move the model and loss to the right device
        if self.use_gpu:
            self.base_model, self.base_loss = copy_model_to_gpu(
                self.base_model, self.base_loss
            )
        else:
            self.base_loss.cpu()
            self.base_model.cpu()

        if self.optimizer is not None:
            self.prepare_optimizer(
                optimizer=self.optimizer, model=self.base_model, loss=self.base_loss
            )

        if self.amp_args is not None:
            if self.amp_type == AmpType.APEX:
                # Initialize apex.amp. This updates the model and the PyTorch optimizer (
                # if training, which is wrapped by the ClassyOptimizer in self.optimizer).
                # Please note this must happen before loading the checkpoint, cause
                # there's amp state to be restored.
                if self.optimizer is None:
                    self.base_model = apex.amp.initialize(
                        self.base_model, optimizers=None, **self.amp_args
                    )
                else:
                    self.base_model, self.optimizer.optimizer = apex.amp.initialize(
                        self.base_model, self.optimizer.optimizer, **self.amp_args
                    )

        if self.simulated_global_batchsize is not None:
            if self.simulated_global_batchsize % self.get_global_batchsize() != 0:
                raise ValueError(
                    f"Global batch size ({self.get_global_batchsize()}) must divide "
                    f"simulated_global_batchsize ({self.simulated_global_batchsize})"
                )
        else:
            self.simulated_global_batchsize = self.get_global_batchsize()

        self.optimizer_period = (
            self.simulated_global_batchsize // self.get_global_batchsize()
        )
        if self.optimizer_period > 1:
            logging.info(
                f"Using gradient accumulation with a period of {self.optimizer_period}"
            )

        if self.checkpoint_path:
            self.checkpoint_dict = load_and_broadcast_checkpoint(self.checkpoint_path)

        classy_state_dict = (
            None
            if self.checkpoint_dict is None
            else self.checkpoint_dict["classy_state_dict"]
        )

        if classy_state_dict is not None:
            state_load_success = update_classy_state(self, classy_state_dict)
            assert (
                state_load_success
            ), "Update classy state from checkpoint was unsuccessful."

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
            self.base_model,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=self.find_unused_parameters,
            bucket_cap_mb=self.ddp_bucket_cap_mb,
        )
        if (
            isinstance(self.base_loss, ClassyLoss)
            and self.base_loss.has_learned_parameters()
        ):
            logging.info("Initializing distributed loss")
            self.distributed_loss = init_distributed_data_parallel_model(
                self.base_loss,
                broadcast_buffers=broadcast_buffers,
                find_unused_parameters=self.find_unused_parameters,
                bucket_cap_mb=self.ddp_bucket_cap_mb,
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

        return where

    def get_classy_state(self, deep_copy: bool = False):
        """Returns serialiable state of task

        Args:
            deep_copy: If true, does a deep copy of state before returning.
        """
        optimizer_state = {}
        if self.optimizer is not None:
            optimizer_state = self.optimizer.get_classy_state()

        classy_state_dict = {
            "train": self.train,
            "base_model": self.base_model.get_classy_state(),
            "meters": [meter.get_classy_state() for meter in self.meters],
            "optimizer": optimizer_state,
            "phase_idx": self.phase_idx,
            "train_phase_idx": self.train_phase_idx,
            "num_updates": self.num_updates,
            "losses": self.losses,
            "hooks": {hook.name(): hook.get_classy_state() for hook in self.hooks},
            "loss": {},
        }
        if "train" in self.datasets and self._is_checkpointable_dataset(
            self.datasets["train"]
        ):
            classy_state_dict["train_dataset_iterator"] = self.datasets[
                "train"
            ].get_classy_state()

        if isinstance(self.base_loss, ClassyLoss):
            classy_state_dict["loss"] = self.base_loss.get_classy_state()
        if self.amp_args is not None:
            if self.amp_type == AmpType.APEX:
                classy_state_dict["amp"] = apex.amp.state_dict()

            elif self.amp_grad_scaler is not None:
                classy_state_dict["amp"] = self.amp_grad_scaler.state_dict()

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
        if self.optimizer is not None:
            self.optimizer.set_classy_state(state["optimizer"])
        if state.get("loss") and isinstance(self.base_loss, ClassyLoss):
            self.base_loss.set_classy_state(state["loss"])

        if "amp" in state:
            if self.amp_type == AmpType.APEX:
                apex.amp.load_state_dict(state["amp"])
            else:
                self.amp_grad_scaler.load_state_dict(state["amp"])

        for hook in self.hooks:
            # we still want to be able to run when new hooks are added or old
            # hooks are removed
            if hook.name() in state["hooks"]:
                hook.set_classy_state(state["hooks"][hook.name()])
            else:
                logging.warn(f"No state found for hook: {hook.name()}")

        if "train" in self.datasets and self._is_checkpointable_dataset(
            self.datasets["train"]
        ):
            self.datasets["train"].set_classy_state(state.get("train_dataset_iterator"))

    @staticmethod
    def _is_checkpointable_dataset(dataset):
        return hasattr(dataset, "get_classy_state") and hasattr(
            dataset, "set_classy_state"
        )

    def eval_step(self):
        self.last_batch = None

        # Process next sample
        with Timer() as timer:
            sample = next(self.data_iterator)

        assert isinstance(sample, dict) and "input" in sample and "target" in sample, (
            f"Returned sample [{sample}] is not a map with 'input' and"
            + "'target' keys"
        )

        target = sample["target"]
        if self.use_gpu:
            sample = recursive_copy_to_gpu(sample, non_blocking=True)

        # Optional Pytorch AMP context
        torch_amp_context = (
            torch.cuda.amp.autocast()
            if self.amp_type == AmpType.PYTORCH
            else contextlib.suppress()
        )

        with torch.no_grad(), torch_amp_context:
            output = self.model(sample["input"])

            local_loss = self.compute_loss(output, sample)

            loss = local_loss.detach().clone()

            self.check_inf_nan(loss)

            self.losses.append(loss.data.cpu().item() * target.size(0))

            self.update_meters(output, sample)

        # Move some data to the task so hooks get a chance to access it
        self.last_batch = LastBatchInfo(
            loss=loss,
            output=output,
            target=target,
            sample=sample,
            step_data={"sample_fetch_time": timer.elapsed_time},
        )

    def check_inf_nan(self, loss):
        if loss == float("inf") or loss == float("-inf") or loss != loss:
            raise FloatingPointError(f"Loss is infinity or NaN: {loss}")

    def train_step(self):
        """Train step to be executed in train loop."""

        self.last_batch = None

        # Process next sample
        with Timer() as timer:
            sample = next(self.data_iterator)

        assert isinstance(sample, dict) and "input" in sample and "target" in sample, (
            f"Returned sample [{sample}] is not a map with 'input' and"
            + "'target' keys"
        )

        # Copy sample to GPU
        target = sample["target"]
        if self.use_gpu:
            sample = recursive_copy_to_gpu(sample, non_blocking=True)

        if self.mixup_transform is not None:
            sample = self.mixup_transform(sample)

        # Optional Pytorch AMP context
        torch_amp_context = (
            torch.cuda.amp.autocast()
            if self.amp_type == AmpType.PYTORCH
            else contextlib.suppress()
        )

        # Forward pass
        with torch.enable_grad(), torch_amp_context:
            output = self.model(sample["input"])

            local_loss = self.compute_loss(output, sample)
            loss = local_loss.detach().clone()
            self.losses.append(loss.data.cpu().item() * target.size(0))

            self.update_meters(output, sample)

        # Backwards pass + optimizer step
        self.run_optimizer(local_loss)

        self.num_updates += self.get_global_batchsize()

        # Move some data to the task so hooks get a chance to access it
        self.last_batch = LastBatchInfo(
            loss=loss,
            output=output,
            target=target,
            sample=sample,
            step_data={"sample_fetch_time": timer.elapsed_time},
        )

    def compute_loss(self, model_output, sample):
        return self.loss(model_output, sample["target"])

    def run_optimizer(self, loss):
        """Runs backwards pass and update the optimizer"""

        self.check_inf_nan(loss)

        # Gradient accumulation logic. We always set optimizer_period, even
        # if gradient accumulation is disabled. Assumes all batches have the
        # same size
        update_idx = self.num_updates // self.get_global_batchsize()
        do_zero_grad = (update_idx % self.optimizer_period) == 0
        do_step = (update_idx % self.optimizer_period) == self.optimizer_period - 1

        if do_zero_grad:
            self.optimizer.zero_grad()

        # only sync with DDP when we need to perform an optimizer step
        ctx_mgr_model = (
            self.distributed_model.no_sync()
            if self.distributed_model is not None and not do_step
            else contextlib.suppress()
        )
        ctx_mgr_loss = (
            self.distributed_loss.no_sync()
            if self.distributed_loss is not None and not do_step
            else contextlib.suppress()
        )

        with ctx_mgr_model, ctx_mgr_loss:
            if self.amp_type == AmpType.APEX:
                with apex.amp.scale_loss(loss, self.optimizer.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.amp_type == AmpType.PYTORCH:
                self.amp_grad_scaler.scale(loss).backward()
            else:
                loss.backward()

        if do_step:
            # Handle gradient accumulation related gradient rescaling
            if self.optimizer_period != 1:
                self._rescale_gradients(1 / self.optimizer_period)

            # Clipping must happen after grad accumulation
            if self.clip_grad_norm is not None:
                self._clip_gradients(self.clip_grad_norm)

            if self.amp_type == AmpType.PYTORCH:
                # If using mixed precision, handle underflow-related scaling
                # See https://pytorch.org/docs/stable/amp.html#gradient-scaling
                # for context
                self.amp_grad_scaler.step(self.optimizer, where=self.where)
                self.amp_grad_scaler.update()
            else:
                self.optimizer.step(where=self.where)

    def _rescale_gradients(self, scale):
        for param in master_params(self.optimizer):
            if param.grad is not None:
                param.grad.data.mul_(scale)

    def _clip_gradients(self, max_norm):
        nn.utils.clip_grad_norm_(master_params(self.optimizer), max_norm)

    def update_meters(self, model_output, sample):
        target = sample["target"].detach().cpu()
        model_output = model_output.detach().cpu()

        # Update meters
        for meter in self.meters:
            meter.update(model_output, target, is_train=self.train)

    def synchronize_losses(self):
        """Average the losses across the different replicas"""

        # Average losses across nodes
        losses_tensor = torch.tensor(self.losses)
        synchronized_losses_tensor = all_reduce_mean(losses_tensor)
        self.losses = synchronized_losses_tensor.tolist()

    def advance_phase(self):
        """Performs bookkeeping / task updates between phases

        Increments phase idx, resets meters, resets loss history,
        resets counters, shuffles dataset, rebuilds iterators, and
        sets the train / test state for phase.
        """
        logging.debug("Advancing phase")
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
        self.build_dataloaders_for_current_phase()
        self.create_data_iterators()
        # Set up pytorch module in train vs eval mode, update optimizer.
        self._set_model_train_mode()

    def done_training(self):
        """Stop condition for training"""
        return self.phase_idx + 1 >= len(self.phases)

    def create_data_iterators(self):
        """Creates data iterator(s) for the current phase."""
        # Delete iterator explicitly so that all dataloader processes
        # are cleaned up.
        del self.data_iterator
        self.data_iterator = iter(self.dataloader)

    def _set_model_train_mode(self):
        """Set train mode for model"""
        phase = self.phases[self.phase_idx]
        self.base_model.train(phase["train"])
        self.base_loss.train(phase["train"])

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
        """Return local replica's batchsize for dataset (e.g. batchsize per GPU)"""
        return self.datasets[self.phase_type].get_batchsize_per_replica()

    def get_global_batchsize(self):
        """Return global batchsize across all trainers"""
        return self.datasets[self.phase_type].get_global_batchsize()

    def on_start(self):
        for hook in self.hooks:
            hook.on_start(self)

    def on_phase_start(self):
        self.phase_start_time_total = time.perf_counter()

        self.advance_phase()

        for hook in self.hooks:
            hook.on_phase_start(self)

        self.phase_start_time_train = time.perf_counter()

    def on_phase_end(self):
        self.log_phase_end("train")

        if self.train:
            self.optimizer.on_epoch(where=self.where)

        logging.debug("Syncing losses on phase end...")
        self.synchronize_losses()
        logging.debug("...losses synced")

        logging.debug("Syncing meters on phase end...")
        for meter in self.meters:
            meter.sync_state()
        logging.debug("...meters synced")
        barrier()

        for hook in self.hooks:
            hook.on_phase_end(self)
        self.perf_log = []

        self.log_phase_end("total")

    def on_end(self):
        for hook in self.hooks:
            hook.on_end(self)

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

    def __repr__(self):
        if hasattr(self, "_config"):
            config = json.dumps(self._config, indent=4)
            return f"{super().__repr__()} initialized with config:\n{config}"

        return super().__repr__()
