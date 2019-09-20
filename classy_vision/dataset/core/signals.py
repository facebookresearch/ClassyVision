#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class Signals(Enum):
    SHUTDOWN_WORKER = -1
    LAST_SAMPLE = -2
    WORKER_DONE = -3
