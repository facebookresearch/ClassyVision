#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Iterable, Iterator

from .dataloader_wrapper import DataloaderWrapper


class DataloaderSkipNoneWrapper(DataloaderWrapper):
    """
    Dataloader which wraps another dataloader and skip `None` batch data.

    Attribute accesses are passed to the wrapped dataloader.
    """

    def __init__(self, dataloader: Iterable) -> None:
        super().__init__(dataloader)
        self.batch_count = 0

    def __iter__(self) -> Iterator[Any]:
        self._iter = iter(self.dataloader)
        return self

    def __next__(self) -> Any:
        # we may get `None` batch data when all the images/videos in the batch
        # are corrupted. In such case, we keep getting the next batch until
        # meeting a good batch.
        next_batch = None
        while next_batch is None:
            next_batch = next(self._iter)

        self.batch_count += 1
        return next_batch

    def __len__(self) -> int:
        # since we do not know the total dataset length ahead of time, we count the
        # batch data as they come in from wrapped dataloader. We expect __len__() is
        # only called after DataloaderSkipNoneWrapper is exhausted.

        try:
            next(self._iter)
            raise AssertionError(
                "Do not call __len__() when the wrapped dataloader is not exhausted yet"
            )
        except StopIteration:
            # Make sure we have exhausted the wrapper dataloader
            logging.info("Return the batch count as the length of dataloader")

        return self.batch_count
