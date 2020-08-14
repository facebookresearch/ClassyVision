#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Iterator

import torch
from classy_vision.generic.util import recursive_copy_to_gpu

from .dataloader_wrapper import DataloaderWrapper


class DataloaderAsyncGPUWrapper(DataloaderWrapper):
    """
    Dataloader which wraps another dataloader, and moves the data to GPU asynchronously.
    At most one batch is pre-emptively copied.

    credits: @vini
    """

    def __init__(self, dataloader: Iterable) -> None:
        super().__init__(dataloader)
        self.cache = None
        self.stream = torch.cuda.Stream()
        assert torch.cuda.is_available(), "This Dataloader wrapper needs a CUDA setup"

    def __iter__(self) -> Iterator[Any]:
        self._iter = iter(self.dataloader)
        return self

    def __next__(self) -> Any:
        result = None

        with torch.cuda.stream(self.stream):
            if self.cache is not None:
                # Make sure that an ongoing transfer is done
                torch.cuda.current_stream().wait_stream(self.stream)
                result = self.cache
            else:
                result = recursive_copy_to_gpu(next(self._iter))

            # Lookahead and start upload
            try:
                self.cache = recursive_copy_to_gpu(next(self._iter))
            except StopIteration:
                self.cache = None
        assert result is not None

        return result

    def __len__(self) -> int:
        return len(self.dataloader)
