#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .checkpoint_hook import CheckpointHook
from .classy_hook import ClassyHook, ClassyHookFunctions
from .exponential_moving_average_model_hook import ExponentialMovingAverageModelHook
from .loss_lr_meter_logging_hook import LossLrMeterLoggingHook
from .model_complexity_hook import ModelComplexityHook
from .model_tensorboard_hook import ModelTensorboardHook
from .profiler_hook import ProfilerHook
from .progress_bar_hook import ProgressBarHook
from .tensorboard_plot_hook import TensorboardPlotHook
from .time_metrics_hook import TimeMetricsHook
from .visdom_hook import VisdomHook


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
