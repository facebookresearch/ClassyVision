#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict


class ClassyTransform(ABC):
    """
    Class representing a data transform abstraction.

    Data transform is most often needed to pre-process input data (e.g. image, video)
    before sending it to a model. But it can also be used for other purposes.
    """

    @abstractmethod
    def __call__(self, image):
        """
        The interface `__call__` is used to transform the input data. It should contain
        the actual implementation of data transform.

        Args:
            image: input image data
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)
