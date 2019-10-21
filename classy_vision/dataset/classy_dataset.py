#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from classy_vision.generic.distributed_util import get_rank, get_world_size
from classy_vision.generic.util import is_pos_int
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _return_true(_sample):
    return True


class ClassyDataset:
    """
    Wrapper to provide typical dataset functionalities in a single place.

    split: String indicating split of dataset to use ("train", "test")
    batchsize_per_replica: Positive int indicating batchsize for training replica
    shuffle: Bool indicating whether we should shuffle between epochs
    transform: Callable, will be applied to each sample
    num_samples: Int, when set restricts the number of samples provided by dataset
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
        batchsize_per_replica = config.get("batchsize_per_replica")
        assert is_pos_int(
            batchsize_per_replica
        ), "batchsize_per_replica must be a positive int"

        shuffle = config.get("use_shuffle")
        assert isinstance(shuffle, bool), "use_shuffle must be a boolean"

        # Num samples is not used in all cases and has a clear default of None
        num_samples = config.get("num_samples", None)
        assert num_samples is None or is_pos_int(
            num_samples
        ), "num_samples must be a positive int"

        transform_config = config.get("transforms")

        return transform_config, batchsize_per_replica, shuffle, num_samples

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self), "Provided idx is outside of dataset range"
        sample = self.dataset[idx]
        if self.transform is None:
            return sample
        return self.transform(sample)

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
