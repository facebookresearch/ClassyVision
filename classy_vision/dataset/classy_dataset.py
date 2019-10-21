#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from classy_vision.dataset.core import Dataset, WrapDataset
from classy_vision.generic.distributed_util import get_rank, get_world_size
from classy_vision.generic.util import is_pos_int
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _return_true(_sample):
    return True


class ClassyDataset(Dataset):
    """
    Interface specifying what a Classy Vision dataset can be expected to provide.

    The main difference between this class and the core dataset class
    is that a classy dataset provides information about the samples /
    targets returned and it has some built in functionality for
    storing configs.
    """

    @classmethod
    def get_available_splits(cls):
        return ["train", "test"]

    def __init__(
        self,
        split: Optional[str],
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Callable],
        num_samples: Optional[int],
    ):
        """
        Classy Dataloader constructor.
        """
        # Assignments:
        self.split = split
        self.batchsize_per_replica = batchsize_per_replica
        self.shuffle = shuffle
        self.transform = transform
        self.num_samples = num_samples
        self.dataset = None

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError()

    @classmethod
    def parse_config(cls, config):
        """
        This function parses out common config options. Those options are

        config: A map with the following string keys:

        batchsize_per_replica: Must be a positive int, batchsize per
        replica (effectively the number of processes performing
        training...usually corresponds to GPU

        use_shuffle: Enable shuffling for the dataset

        num_samples: Artificially restrict the number of samples in a dataset epoch

        transforms (optional): list of tranform configurations to be applied in order
        """
        assert "batchsize_per_replica" in config and is_pos_int(
            config["batchsize_per_replica"]
        ), "batchsize_per_replica must be a positive int"

        assert "use_shuffle" in config and isinstance(
            config["use_shuffle"], bool
        ), "use_shuffle must be a boolean"

        assert "num_samples" in config and (
            config["num_samples"] is None or is_pos_int(config["num_samples"])
        ), "num_samples must be a positive int"

        transform_config = config.get("transforms")
        shuffle = config.get("use_shuffle")
        num_samples = config.get("num_samples", None)

        return transform_config, config["batchsize_per_replica"], shuffle, num_samples

    @classmethod
    def wrap_dataset(
        cls,
        dataset,
        transform=None,
        batchsize_per_replica=None,  # Unused
        filter_func=_return_true,  # Unused
        shuffle=True,  # Unused
        subsample=None,  # Unused
        shard_group_size=1,  # Unused
    ):
        """
        WARNING: This function is going to be deleted as part of the
        migration of the Classy Vision datasets to canonical pytorch
        datasets. If you see this function in your codebase, you have
        gotten an intermediate dataset state. This state should still
        be functional, but if you rebase this function will disappear

        Wraps self.dataset with TransformDataset..

        If this is not a distributed run, we still wrap the dataset in
        shard dataset, but with world size 1 and rank 0.

        This can only be called once, preferably during construction.
        """
        # apply all the transformations needed:
        if not isinstance(dataset, Dataset):
            dataset = WrapDataset(dataset)

        # Apply transforms
        if transform is not None:
            dataset = dataset.transform(transform)

        return dataset

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self), "Provided idx is outside of dataset range"
        return self.dataset[idx]

    def __len__(self):
        assert self.num_samples is None or self.num_samples <= len(
            self.dataset
        ), "Num samples mus be less than length of base dataset"
        return len(self.dataset) if self.num_samples is None else self.num_samples

    def _get_sampler(self):
        world_size = get_world_size()
        rank = get_rank()
        return DistributedSampler(
            self, num_replicas=world_size, rank=rank, shuffle=self.shuffle
        )

    def iterator(self, *args, **kwargs):
        return DataLoader(
            self,
            batch_size=self.batchsize_per_replica,
            num_workers=kwargs.get("num_workers", 0),
            pin_memory=kwargs.get("pin_memory", False),
            multiprocessing_context=kwargs.get("multiprocessing_context", None),
            sampler=self._get_sampler(),
        )

    def get_batchsize_per_replica(self):
        return self.batchsize_per_replica

    def get_global_batchsize(self):
        return self.get_batchsize_per_replica() * get_world_size()

    def get_classy_state(self):
        """Get state for object (e.g. shuffle)"""
        return {
            "split": self.split,
            "batchsize_per_replica": self.batchsize_per_replica,
            "shuffle": self.shuffle,
            "num_samples": self.num_samples,
            "state": {"dataset_type": type(self)},
            "wrapped_state": self.dataset.get_classy_state()
            if self.dataset is not None
            else None,
        }

    def set_classy_state(self, state, strict=True):
        """Sets state for object (e.g. shuffle)"""
        assert isinstance(self, state["state"]["dataset_type"]) or not strict, (
            "Type of saved state does not match current object. "
            "If intentional, use non-strict flag"
        )
        self.split = state["split"]
        self.batchsize_per_replica = state["batchsize_per_replica"]
        self.shuffle = state["shuffle"]
        self.num_samples = state["num_samples"]
        if self.dataset is not None:
            self.dataset.set_classy_state(state["wrapped_state"])
