#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
            Dataset is first divided into groups of samples. Sharding is done by
            going over the original dataset by taking `mod` operation to sample
            groups.  (detailed example below). For incomplete shards, roll over
            the real samples to generate dummy samples.

            Example:
                dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                world_size: 4

            when group_size = 1
                    RANK    |  shard_dataset  | is_dummy_sample
                    ===========================================
                    rank_0  |  [0, 4, 8, 12]  |  [0, 0, 0, 0]
                    rank_1  |  [1, 5, 9, 13]  |  [0, 0, 0, 0]
                    rank_2  |  [2, 6, 10, 2]  |  [0, 0, 0, 1]
                    rank_3  |  [3, 7, 11, 3]  |  [0, 0, 0, 1]

            when group_size = 2

                    RANK    |  shard_dataset  | is_dummy_sample
                    ===========================================
                    rank_0  |  [0, 1, 8, 9]  |  [0, 0, 0, 0]
                    rank_1  |  [2, 3, 10, 11]  |  [0, 0, 0, 0]
                    rank_2  |  [4, 5, 4, 5]  |  [0, 0, 1, 1]
                    rank_3  |  [6, 7, 6, 7]  |  [0, 0, 1, 1]


        Note:
            shard_dataset operates only on dataset with individual samples and
            not batch_dataset.
            .batch() call on dataset must happen after .shard() call.
    """

    def __init__(self, dataset, world_size, rank, group_size=1):
        assert len(dataset) % group_size == 0, (
            "dataset length must be a multiplier of group size"
            "dataset length: %d, group size: %d" % (len(dataset), group_size)
        )
        self.world_size = world_size
        self.rank = rank
        self.dataset = dataset
        self.group_size = group_size
        dataset_group_size = len(self.dataset) // group_size
        self.real_shard_length = (
            (dataset_group_size // self.world_size)
            + (self.rank < (dataset_group_size % self.world_size))
        ) * self.group_size

    def __getitem__(self, idx):
        # For last shard, loop over the real samples to generate dummy samples.
        is_dummy_sample = 0
        if idx >= self.real_shard_length:
            is_dummy_sample = 1
        is_dummy_sample = torch.tensor(is_dummy_sample)
        idx = idx % self.real_shard_length
        group_idx = idx // self.group_size
        group_remainder = idx % self.group_size
        dataset_index = (
            int(group_idx * self.world_size + self.rank) * self.group_size
            + group_remainder
        )

        sample = self.dataset[dataset_index]
        assert sample is None or isinstance(
            sample, dict
        ), "Shard dataset only supports None / dict samples"
        # If sample is not None or empty dict
        if sample is not None and len(sample) != 0:
            sample["is_dummy_sample"] = is_dummy_sample
        return sample

    def __len__(self):
        dataset_group_size = len(self.dataset) // self.group_size
        return int(math.ceil(dataset_group_size / self.world_size)) * self.group_size
