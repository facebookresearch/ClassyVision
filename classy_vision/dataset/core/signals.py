#!/usr/bin/env python3

from enum import Enum


class Signals(Enum):
    SHUTDOWN_WORKER = -1
    LAST_SAMPLE = -2
    WORKER_DONE = -3
