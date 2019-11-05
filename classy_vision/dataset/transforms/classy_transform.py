#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict


class ClassyTransform(ABC):
    @abstractmethod
    def __call__(self, image):
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)
