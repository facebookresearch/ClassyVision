#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator


class DataloaderWrapper(ABC):
    """
    Abstract class representing dataloader which wraps another dataloader.

    Attribute accesses are passed to the wrapped dataloader.
    """

    def __init__(self, dataloader: Iterable) -> None:
        # we use self.__dict__ to set the attributes since the __setattr__ method
        # is overridden
        attributes = {"dataloader": dataloader, "_iter": None}
        self.__dict__.update(attributes)

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        pass

    @abstractmethod
    def __next__(self) -> Any:
        pass

    def __getattr__(self, attr) -> Any:
        """
        Pass the getattr call to the wrapped dataloader
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        return getattr(self.dataloader, attr)

    def __setattr__(self, attr, value) -> None:
        """
        Pass the setattr call to the wrapped dataloader
        """
        if attr in self.__dict__:
            self.__dict__[attr] = value
        else:
            setattr(self.dataloader, attr, value)
