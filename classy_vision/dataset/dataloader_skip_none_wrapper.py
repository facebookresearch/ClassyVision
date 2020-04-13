#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Iterator

from .dataloader_wrapper import DataloaderWrapper


class DataloaderSkipNoneWrapper(DataloaderWrapper):
    """
    Dataloader which wraps another dataloader and skip `None` batch data.

    Attribute accesses are passed to the wrapped dataloader.
    """

    def __init__(self, dataloader: Iterable) -> None:
        super().__init__(dataloader)

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
        return next_batch
