#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .backfill_async_dataset_iterator import _BackfillAsyncDatasetIterator
from .base_async_dataset_iterator import _BaseAsyncDatasetIterator
from .dataset_iterator import DatasetIterator


class AsyncDatasetIterator(DatasetIterator):
    """
    Returns an asynchronous iterator over a dataset.  Supports
    backfilling failed / corrupted images at the expense of
    maintaining dataset order.

    This iterator is much simpler than the PyTorch one: the only "extra"
    option it has is to pin memory (as this cannot happen in work process).

    If mp_start_method is None, it defaults to current context. If
    set, then the dataloader will create new processes using the
    specified method: fork, spawn, forkserver.
    """

    def __init__(
        self,
        dataset,
        num_workers=0,
        pin_memory=False,
        seed=0,
        backfill_batches=False,
        mp_start_method=None,
    ):
        super(AsyncDatasetIterator, self).__init__(dataset)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.backfill_batches = backfill_batches
        self.mp_start_method = mp_start_method

    def __iter__(self):
        if self.backfill_batches:
            return _BackfillAsyncDatasetIterator(
                self.dataset,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                seed=self.seed,
                mp_start_method=self.mp_start_method,
            )
        return _BaseAsyncDatasetIterator(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            seed=self.seed,
            mp_start_method=self.mp_start_method,
        )

    def __len__(self):
        return len(self.dataset)
