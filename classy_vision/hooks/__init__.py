#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules


# The order of imports matters here because of a circular dependency. constants
# must come before any hook
from .constants import ClassyHookFunctions  # isort:skip
from .checkpoint_hook import CheckpointHook  # isort:skip
from .classy_hook import ClassyHook  # isort:skip
from .exponential_moving_average_model_hook import (  # isort:skip
    ExponentialMovingAverageModelHook,
)
from .loss_lr_meter_logging_hook import LossLrMeterLoggingHook  # isort:skip
from .model_complexity_hook import ModelComplexityHook  # isort:skip
from .model_tensorboard_hook import ModelTensorboardHook  # isort:skip
from .profiler_hook import ProfilerHook  # isort:skip
from .progress_bar_hook import ProgressBarHook  # isort:skip
from .tensorboard_plot_hook import TensorboardPlotHook  # isort:skip
from .time_metrics_hook import TimeMetricsHook  # isort:skip
from .visdom_hook import VisdomHook  # isort:skip


__all__ = [
    "CheckpointHook",
    "ClassyHook",
    "ClassyHookFunctions",
    "ExponentialMovingAverageModelHook",
    "LossLrMeterLoggingHook",
    "TensorboardPlotHook",
    "ModelComplexityHook",
    "ModelTensorboardHook",
    "ProfilerHook",
    "ProgressBarHook",
    "TimeMetricsHook",
    "VisdomHook",
]

FILE_ROOT = Path(__file__).parent

# automatically import any Python files in the hooks/ directory
import_all_modules(FILE_ROOT, "classy_vision.hooks")
