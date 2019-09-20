#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from classy_vision.dataset.core import Dataset, WrapDataset
from classy_vision.generic.distributed_util import get_rank, get_world_size
from classy_vision.generic.util import is_pos_int


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

    def __init__(self, config):
        """
        Classy Dataloader constructor.
        """
        # Assignments:
        self._config = config
        self._split = config["split"] if "split" in config else None
        self.dataset = None

    def parse_config(self, config):
        """
        This function parses out common config options. Those options are

        config: A map with the following string keys:

        batchsize_per_replica: Must be a positive int, batchsize per
        replica (effectively the number of processes performing
        training...usually corresponds to GPU

        use_shuffle: Enable shuffling for the dataset (adds "do_shuffle"
        member for shuffling on command)

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

    def wrap_dataset(
        self,
        dataset,
        transform=None,
        batchsize_per_replica=1,
        filter_func=_return_true,
        shuffle=True,
        subsample=None,
    ):
        """
        Wraps self.dataset with TransformDataset, ShuffleDataset,
        ResampleDataset, ShardDataset and BatchDataset. For the filter
        function, True = keep sample.

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

        # Apply shuffle
        if shuffle:
            dataset = dataset.shuffle()

        # subsample dataset (if requested):
        if subsample is not None and subsample:
            dataset = dataset.resample([n for n in range(subsample)])

        # shard data
        dataset = dataset.shard(get_world_size(), get_rank())

        # batch data if requested:
        dataset = dataset.batch(batchsize_per_replica, filter_func=filter_func)

        return dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_batchsize_per_replica(self):
        # this searches for batchsize_per_replica in self and then in self.dataset
        return getattr(self, "batchsize_per_replica", 1)

    def get_global_batchsize(self):
        return self.get_batchsize_per_replica() * get_world_size()

    def get_classy_state(self):
        """Get state for object (e.g. shuffle)"""
        return {
            "config": self._config,
            "split": self._split,
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
        self._config = state["config"]
        self._split = state["split"]
        if self.dataset is not None:
            self.dataset.set_classy_state(state["wrapped_state"])
