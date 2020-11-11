#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from classy_vision.generic.util import get_batchsize_per_replica, recursive_copy_to_gpu
from classy_vision.hooks import ClassyHook, register_hook
from fvcore.nn.precise_bn import update_bn_stats


def _get_iterator(data_iter, use_gpu):
    for elem in data_iter:
        if use_gpu:
            elem = recursive_copy_to_gpu(elem, non_blocking=True)
        yield elem["input"]


@register_hook("precise_bn")
class PreciseBatchNormHook(ClassyHook):
    """Hook to compute precise batch normalization statistics.

    Batch norm stats are calculated and updated during training, when the weights are
    also changing, which makes the calculations imprecise. This hook recomputes the
    batch norm stats at the end of a train phase to make them more precise. See
    `fvcore's documentation <https://github.com/facebookresearch/fvcore/blob/master/
    fvcore/nn/precise_bn.py>`_ for more information.
    """

    on_phase_start = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, num_samples: int) -> None:
        """The constructor method of PreciseBatchNormHook.

        Caches the required number of samples on the CPU during train phases

        Args:
            num_samples: Number of samples to calculate the batch norm stats per replica
        """
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples has to be a positive integer")
        self.num_samples = num_samples
        self.batch_size = None

    @classmethod
    def from_config(cls, config):
        return cls(config["num_samples"])

    def on_start(self, task) -> None:
        logging.info("Use precise BatchNorm hook")

    def on_step(self, task) -> None:
        if not task.train or self.batch_size is not None:
            return

        self.batch_size = get_batchsize_per_replica(task.last_batch.sample["input"])

    def on_phase_end(self, task) -> None:
        if not task.train:
            return

        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        task.build_dataloaders_for_current_phase()
        task.create_data_iterators()
        if num_batches > len(task.data_iterator):
            num_batches = len(task.data_iterator)
            logging.info(f"Reduce no. of samples to {num_batches * self.batch_size}")

        iterator = _get_iterator(task.data_iterator, task.use_gpu)
        update_bn_stats(task.base_model, iterator, num_batches)
