#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.config_utils import get_test_task_config

from classy_vision.tasks import build_task


class TestDataloaderLimitWrapper(unittest.TestCase):
    def _test_number_of_batches(self, data_iterator, expected_batches):
        num_batches = 0
        for _ in data_iterator:
            num_batches += 1
        self.assertEqual(num_batches, expected_batches)

    def test_streaming_dataset(self):
        """
        Test that streaming datasets return the correct number of batches, and that
        the length is also calculated correctly.
        """
        config = get_test_task_config()
        dataset_config = {
            "name": "synthetic_image_streaming",
            "split": "train",
            "crop_size": 224,
            "class_ratio": 0.5,
            "num_samples": 2000,
            "length": 4000,
            "seed": 0,
            "batchsize_per_replica": 32,
            "use_shuffle": True,
        }
        expected_batches = 62
        config["dataset"]["train"] = dataset_config
        task = build_task(config)
        task.prepare()
        task.advance_phase()
        # test that the number of batches expected is correct
        self.assertEqual(task.num_batches_per_phase, expected_batches)

        # test that the data iterator returns the expected number of batches
        data_iterator = task.get_data_iterator()
        self._test_number_of_batches(data_iterator, expected_batches)

        # test that the dataloader can be rebuilt from the dataset inside it
        task._recreate_data_loader_from_dataset()
        task.create_data_iterator()
        data_iterator = task.get_data_iterator()
        self._test_number_of_batches(data_iterator, expected_batches)
