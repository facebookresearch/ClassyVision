#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Sequence, Union

from classy_vision.dataset.transforms import ClassyTransform
from classy_vision.generic.distributed_util import get_rank, get_world_size
from classy_vision.generic.util import is_pos_int, log_class_usage
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _return_true(_sample):
    return True


DEFAULT_NUM_WORKERS = 4


class ClassyDataset:
    """
    Class representing a dataset abstraction.

    This class wraps a :class:`torch.utils.data.Dataset` via the `dataset` attribute
    and configures the dataloaders needed to access the datasets. By default,
    this class will use `DEFAULT_NUM_WORKERS` processes to load the data
    (num_workers in :class:`torch.utils.data.DataLoader`).
    Transforms which need to be applied to the data should be specified in this class.
    ClassyDataset can be instantiated from a configuration file as well.
    """

    def __init__(
        self,
        dataset: Sequence,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Union[ClassyTransform, Callable]],
        num_samples: Optional[int],
    ) -> None:
        """
        Constructor for a ClassyDataset.

        Args:
            batchsize_per_replica: Positive integer indicating batch size for each
                replica
            shuffle: Whether to shuffle between epochs
            transform: When set, transform to be applied to each sample
            num_samples: When set, this restricts the number of samples provided by
                the dataset
        """
        # Asserts:
        assert is_pos_int(
            batchsize_per_replica
        ), "batchsize_per_replica must be a positive int"
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        assert num_samples is None or is_pos_int(
            num_samples
        ), "num_samples must be a positive int or None"

        # Assignments:
        self.batchsize_per_replica = batchsize_per_replica
        self.shuffle = shuffle
        self.transform = transform
        self.num_samples = num_samples
        self.dataset = dataset
        self.num_workers = DEFAULT_NUM_WORKERS

        log_class_usage("Dataset", self.__class__)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyDataset":
        """Instantiates a ClassyDataset from a configuration.

        Args:
            config: A configuration for the ClassyDataset.

        Returns:
            A ClassyDataset instance.
        """
        raise NotImplementedError

    @classmethod
    def parse_config(cls, config: Dict[str, Any]):
        """
        This function parses out common config options.

        Args:
            config: A dict with the following string keys -

                | *batchsize_per_replica* (int): Must be a positive int, batch size
                |    for each replica
                | *use_shuffle* (bool): Whether to enable shuffling for the dataset
                | *num_samples* (int, optional): When set, restricts the number of
                     samples in a dataset
                | *transforms*: list of tranform configurations to be applied in order

        Returns:
            A tuple containing the following variables -
                | *transform_config*: Config for the dataset transform. Can be passed to
                |    :func:`transforms.build_transform`
                | *batchsize_per_replica*: Batch size per replica
                | *shuffle*: Whether we should shuffle between epochs
                | *num_samples*: When set, restricts the number of samples in a dataset
        """
        batchsize_per_replica = config.get("batchsize_per_replica")
        shuffle = config.get("use_shuffle")
        num_samples = config.get("num_samples")
        transform_config = config.get("transforms")
        return transform_config, batchsize_per_replica, shuffle, num_samples

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def __getitem__(self, idx: int):
        assert idx >= 0 and idx < len(
            self.dataset
        ), "Provided idx ({}) is outside of dataset range".format(idx)
        sample = self.dataset[idx]
        if self.transform is None:
            return sample
        return self.transform(sample)

    def __len__(self):
        assert self.num_samples is None or self.num_samples <= len(
            self.dataset
        ), "Num samples mus be less than length of base dataset"
        return len(self.dataset) if self.num_samples is None else self.num_samples

    def _get_sampler(self, epoch: int):
        """
        Return a :class:`torch.utils.data.sampler.Sampler` to sample the data.

        This is used to distribute the data across the replicas. If shuffling
        is enabled, every epoch will have a different shuffle.

        Args:
            epoch: The epoch being fetched.

        Returns:
            A sampler which tells the data loader which sample to load next.
        """
        world_size = get_world_size()
        rank = get_rank()
        sampler = DistributedSampler(
            self, num_replicas=world_size, rank=rank, shuffle=self.shuffle
        )
        sampler.set_epoch(epoch)
        return sampler

    def iterator(self, *args, **kwargs):
        """
        Returns an iterable which can be used to iterate over the data.

        Args:
            shuffle_seed (int, optional): Seed for the shuffle
            current_phase_id (int, optional): The epoch being fetched. Needed so that
                each epoch has a different shuffle order
        Returns:
            An iterable over the data
        """
        # TODO: Fix naming to be consistent (i.e. everyone uses epoch)
        shuffle_seed = kwargs.get("shuffle_seed", 0)
        assert isinstance(shuffle_seed, int), "Shuffle seed must be an int"
        epoch = kwargs.get("current_phase_id", 0)
        assert isinstance(epoch, int), "Epoch must be an int"
        num_workers_override = kwargs.get("num_workers", self.num_workers)
        if num_workers_override == 0:
            # set the mp context to None to placate the PyTorch dataloader
            kwargs["multiprocessing_context"] = None

        offset_epoch = shuffle_seed + epoch

        return DataLoader(
            self,
            batch_size=self.batchsize_per_replica,
            num_workers=num_workers_override,
            pin_memory=kwargs.get("pin_memory", False),
            worker_init_fn=kwargs.get("worker_init_fn", None),
            multiprocessing_context=kwargs.get("multiprocessing_context", None),
            sampler=self._get_sampler(epoch=offset_epoch),
        )

    def get_batchsize_per_replica(self):
        """
        Get the batch size per replica.

        Returns:
            The batch size for each replica.
        """
        return self.batchsize_per_replica

    def get_global_batchsize(self):
        """
        Get the global batch size, combined over all the replicas.

        Returns:
            The overall batch size of the dataset.
        """
        return self.get_batchsize_per_replica() * get_world_size()
