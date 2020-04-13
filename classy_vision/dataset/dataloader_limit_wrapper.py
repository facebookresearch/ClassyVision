#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Iterable, Iterator

from .dataloader_wrapper import DataloaderWrapper


class DataloaderLimitWrapper(DataloaderWrapper):
    """
    Dataloader which wraps another dataloader and only returns a limited
    number of items.

    This is useful for Iterable datasets where the length of the datasets isn't known.
    Such datasets can wrap their returned iterators with this class. See
    :func:`SyntheticImageStreamingDataset.iterator` for an example.

    Attribute accesses are passed to the wrapped dataloader.
    """

    def __init__(
        self, dataloader: Iterable, limit: int, wrap_around: bool = True
    ) -> None:
        """Constructor for DataloaderLimitWrapper.

        Args:
            dataloader: The dataloader to wrap around
            limit: Specify the number of calls to the underlying dataloader. The wrapper
                will raise a `StopIteration` after `limit` calls.
            wrap_around: Whether to wrap around the original datatloader if the
                dataloader is exhausted before `limit` calls.
        Raises:
            RuntimeError: If `wrap_around` is set to `False` and the underlying
                dataloader is exhausted before `limit` calls.
        """
        super().__init__(dataloader)
        # we use self.__dict__ to set the attributes since the __setattr__ method
        # is overridden
        attributes = {"limit": limit, "wrap_around": wrap_around, "_count": None}
        self.__dict__.update(attributes)

    def __iter__(self) -> Iterator[Any]:
        self._iter = iter(self.dataloader)
        self._count = 0
        return self

    def __next__(self) -> Any:
        if self._count >= self.limit:
            raise StopIteration
        self._count += 1
        try:
            return next(self._iter)
        except StopIteration:
            if self.wrap_around:
                # create a new iterator to load data from the beginning
                logging.info(
                    f"Wrapping around after {self._count} calls. Limit: {self.limit}"
                )
                try:
                    self._iter = iter(self.dataloader)
                    return next(self._iter)
                except StopIteration:
                    raise RuntimeError(
                        "Looks like the dataset is empty, "
                        "have you configured it properly?"
                    )
            else:
                raise RuntimeError(
                    f"StopIteration raised before {self.limit} items were returned"
                )

    def __len__(self) -> int:
        return self.limit
