#!/usr/bin/env python3

import math

import torch

from .dataset import Dataset


class ShardDataset(Dataset):
    """
        Dataset that shards a dataset based on the distributed world
        size and rank of the current worker. rank must be in the range of
        [0, world_size).

        If the dataset size is not divisible by world_size, then for the remaining
        shard, use a dummy sample looping over from the shard samples.
        Meters and loss from dummy_sample will be ignored.

        Sharding logic:
        Sharding is done by going over the original dataset by taking `mod` operation
        (detailed example below).
        For incomplete shards, roll over the real samples to generate dummy samples.

        Example:
            dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            world_size: 4

            RANK    |  shard_dataset  | is_dummy_sample
            ===========================================
            rank_0  |  [0, 4, 8, 12]  |  [0, 0, 0, 0]
            rank_1  |  [1, 5, 9, 13]  |  [0, 0, 0, 0]
            rank_2  |  [2, 6, 10, 2]  |  [0, 0, 0, 1]
            rank_3  |  [3, 7, 11, 3]  |  [0, 0, 0, 1]

        Note:
            shard_dataset operates only on dataset with individual samples and
            not batch_dataset.
            .batch() call on dataset must happen after .shard() call.
    """

    def __init__(self, dataset, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.dataset = dataset
        self.real_shard_length = (len(self.dataset) // self.world_size) + (
            self.rank < (len(self.dataset) % self.world_size)
        )

    def __getitem__(self, idx):
        # For last shard, loop over the real samples to generate dummy samples.
        is_dummy_sample = 0
        if idx >= self.real_shard_length:
            is_dummy_sample = 1
        is_dummy_sample = torch.tensor(is_dummy_sample)
        idx = idx % self.real_shard_length
        dataset_index = int(idx * self.world_size + self.rank)
        sample = self.dataset[dataset_index]
        assert sample is None or isinstance(
            sample, dict
        ), "Shard dataset only supports None / dict samples"
        # If sample is not None or empty dict
        if sample is not None and len(sample) != 0:
            sample["is_dummy_sample"] = is_dummy_sample
        return sample

    def __len__(self):
        return int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
