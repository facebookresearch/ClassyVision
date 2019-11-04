#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from classy_vision.generic.distributed_util import set_cpu_device, set_cuda_device_index

from .classy_trainer import ClassyTrainer


class LocalTrainer(ClassyTrainer):
    def __init__(self, use_gpu=None, num_dataloader_workers=0):
        super().__init__(use_gpu=use_gpu, num_dataloader_workers=num_dataloader_workers)
        if self.use_gpu:
            logging.info("Using GPU, CUDA device index: {}".format(0))
            set_cuda_device_index(0)
        else:
            logging.info("Using CPU")
            set_cpu_device()
