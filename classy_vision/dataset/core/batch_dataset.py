#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch.utils.data.dataloader import default_collate

from .dataset import Dataset


def _return_true(sample):
    return True


class BatchDataset(Dataset):
    """
        Dataset that performs batching of another dataset.

        filter_func skips samples.
    """

    def __init__(
        self, dataset, batchsize_per_replica, filter_func=_return_true, skip_last=False
    ):

        # assertions:
        super(BatchDataset, self).__init__()
        assert isinstance(dataset, Dataset)
        assert isinstance(batchsize_per_replica, int) and batchsize_per_replica >= 1
        assert isinstance(skip_last, bool)
        assert callable(filter_func)

        # store class variables:
        self.dataset = dataset
        self.batchsize_per_replica = batchsize_per_replica
        self.skip_last = skip_last
        self.filter_func = filter_func

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self)

        # get samples to go in batch:
        batch = []
        start_idx = idx * self.batchsize_per_replica
        end_idx = min((idx + 1) * self.batchsize_per_replica, len(self.dataset))
        for n in range(start_idx, end_idx):
            sample = self.dataset[n]
            if not self.filter_func(sample):
                continue
            if torch.is_tensor(sample):
                sample = sample
            batch.append(sample)
        # batch data:
        batch = default_collate(batch)
        return batch

    def __len__(self):
        base_size = float(len(self.dataset)) / float(self.batchsize_per_replica)
        return int(math.floor(base_size) if self.skip_last else math.ceil(base_size))
