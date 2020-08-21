#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Iterator

import torch
from classy_vision.generic.util import recursive_copy_to_gpu

from .dataloader_wrapper import DataloaderWrapper


# See Nvidia's data_prefetcher for reference
# https://github.com/NVIDIA/apex/blob/2ca894da7be755711cbbdf56c74bb7904bfd8417/examples/imagenet/main_amp.py#L264


class DataloaderAsyncGPUWrapper(DataloaderWrapper):
    """
    Dataloader which wraps another dataloader, and moves the data to GPU asynchronously.
    At most one batch is pre-emptively copied (per worker).

    credits: @vini, nvidia Apex
    """

    def __init__(self, dataloader: Iterable) -> None:
        assert torch.cuda.is_available(), "This Dataloader wrapper needs a CUDA setup"

        super().__init__(dataloader)
        self.cache = None
        self.cache_next = None
        self.stream = torch.cuda.Stream()
        self._iter = None

    def __iter__(self) -> Iterator[Any]:
        # The wrapped dataloader may have been changed in place
        # rebuild a new iterator and prefetch
        self._iter = iter(self.dataloader)
        self.preload()
        return self

    def preload(self):
        # Get data from the iterator
        try:
            self.cache_next = next(self._iter)

            # Copy to the device, in a parallel CUDA stream
            with torch.cuda.stream(self.stream):
                self.cache = recursive_copy_to_gpu(self.cache_next, non_blocking=True)

        except StopIteration:
            self.cache = None
            return

    def __next__(self) -> Any:
        # Make sure that future work in the main stream (training loop for instance)
        # waits for the dependent self.stream to be done
        torch.cuda.current_stream().wait_stream(self.stream)

        result = self.cache
        if self.cache is None:
            raise StopIteration

        # Pre-load the next sample
        self.preload()

        return result

    def __len__(self) -> int:
        return len(self.dataloader)
