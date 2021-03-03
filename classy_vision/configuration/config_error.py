# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List


class ConfigError(Exception):
    pass


class ConfigUnusedKeysError(ConfigError):
    def __init__(self, unused_keys: List[str]):
        self.unused_keys = unused_keys
        super().__init__(f"The following keys were unused: {self.unused_keys}")
