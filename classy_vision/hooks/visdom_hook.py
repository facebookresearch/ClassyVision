#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
from typing import Any, Dict

from classy_vision.generic.distributed_util import is_primary
from classy_vision.generic.util import flatten_dict
from classy_vision.generic.visualize import plot_learning_curves
from classy_vision.hooks import register_hook
from classy_vision.hooks.classy_hook import ClassyHook


try:
    from visdom import Visdom

    visdom_available = True
except ImportError:
    visdom_available = False


@register_hook("visdom")
class VisdomHook(ClassyHook):
    """Plots metrics on to `Visdom <https://github.com/facebookresearch/visdom>`_.

    Visdom is a flexible tool for creating, organizing, and sharing visualizations
        of live, rich data. It supports Python.

    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_step = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(
        self, server: str, port: str, env: str = "main", title_suffix: str = ""
    ) -> None:
        """
        Args:
            server: host name of the visdom server
            port: port of visdom server, such as 8097
            env: environment of visdom
            title_suffix: suffix that will be appended to the title
        """
        super().__init__()
        if not visdom_available:
            raise RuntimeError("Visdom is not installed, cannot use VisdomHook")

        self.server: str = server
        self.port: str = port
        self.env: str = env
        self.title_suffix: str = title_suffix

        self.metrics: Dict = {}
        self.visdom: Visdom = Visdom(self.server, self.port)

    def on_phase_end(self, task) -> None:
        """
        Plot the metrics on visdom.
        """
        phase_type = task.phase_type
        metrics = self.metrics
        batches = len(task.losses)

        if batches == 0:
            return

        # Loss for the phase
        loss = sum(task.losses) / (batches * task.get_batchsize_per_replica())
        loss_key = phase_type + "_loss"
        if loss_key not in metrics:
            metrics[loss_key] = []
        metrics[loss_key].append(loss)

        # Optimizer LR for the phase
        optimizer_lr = task.optimizer.options_view.lr
        lr_key = phase_type + "_learning_rate"
        if lr_key not in metrics:
            metrics[lr_key] = []
        metrics[lr_key].append(optimizer_lr)

        # Calculate meters
        for meter in task.meters:
            if isinstance(meter.value, collections.MutableMapping):
                flattened_meters_dict = flatten_dict(meter.value, prefix=meter.name)
                for k, v in flattened_meters_dict.items():
                    metric_key = phase_type + "_" + k
                    if metric_key not in metrics:
                        metrics[metric_key] = []
                    metrics[metric_key].append(v)
            else:
                metric_key = phase_type + "_" + meter.name
                if metric_key not in metrics:
                    metrics[metric_key] = []
                metrics[metric_key].append(meter.value)

        # update learning curve visualizations:
        phase_type = "train" if task.train else "test"
        title = "%s-%s" % (phase_type, task.base_model.__class__.__name__)
        title += self.title_suffix

        if not task.train and is_primary():
            logging.info("Plotting learning curves to visdom")
            plot_learning_curves(
                metrics, visdom_server=self.visdom, env=self.env, win=title, title=title
            )
