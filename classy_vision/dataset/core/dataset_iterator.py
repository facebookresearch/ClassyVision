#!/usr/bin/env python3

import torch

from .backfill_async_dataset_iterator import (
    backfill_batch,
    recursive_batchsize_per_replica,
)
from .dataset import Dataset


class _DatasetIterator(object):
    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.idx = 0

    def __next__(self):
        if self.idx < len(self.dataset):
            current_idx = self.idx
            self.idx += 1
            sample = self.dataset[current_idx]
            if sample is None:  # dataset can return None to signal end of data
                raise StopIteration()
            return sample
        else:
            raise StopIteration()

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset)


class _BackfillDatasetIterator(object):
    def __init__(self, dataset):
        assert (
            hasattr(dataset, "batchsize_per_replica")
            and dataset.batchsize_per_replica > 0
        ), """The backfill iterators can only be used
            with datasets with a valid batchsize_per_replica"""
        self.dataset = dataset
        self.idx = 0
        self.unfinished_batch = None

    def __next__(self):
        batch = None
        while self.idx < len(self.dataset):
            next_batch = self.dataset[self.idx]
            self.idx += 1

            # backfill_batch returns two lists
            batch, self.unfinished_batch = backfill_batch(
                self.unfinished_batch, next_batch, self.dataset.batchsize_per_replica
            )

            if batch is not None:
                return batch

        if (
            self.unfinished_batch is not None
            and recursive_batchsize_per_replica(self.unfinished_batch) > 0
        ):
            last_batch = self.unfinished_batch
            self.unfinished_batch = None
            return last_batch

        raise StopIteration


class DatasetIterator(object):
    """
        Synchronous iterator over a dataset.
    """

    def __init__(self, dataset, backfill_batches=False):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.backfill_batches = backfill_batches

    def __iter__(self):
        if self.backfill_batches:
            return _BackfillDatasetIterator(self.dataset)

        return _DatasetIterator(self.dataset)

    def __len__(self):
        return len(self.dataset)
