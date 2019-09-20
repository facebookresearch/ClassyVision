#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.utils import create_test_dataset

import torch
from classy_vision.dataset.core.shard_dataset import ShardDataset


class ShardDatasetTest(unittest.TestCase):
    def test_shard_dataset_length(self):
        """
        Test that sharded dataset length is correct.
        """
        test_dataset, _ = create_test_dataset(tensor_size=(14, 3, 224, 224))
        WORLD_SIZE = 4

        for rank in range(WORLD_SIZE):
            shard_dataset = ShardDataset(test_dataset, world_size=WORLD_SIZE, rank=rank)
            self.assertTrue(len(shard_dataset) == 4)

    def test_non_final_shard_dataset_index(self):
        """
        Test that for non-final shard, the indices map to
        correct indices from the main dataset.
        """
        test_dataset, _ = create_test_dataset(tensor_size=(14, 3, 224, 224))
        WORLD_SIZE = 4
        RANK = 1
        shard_dataset = ShardDataset(test_dataset, world_size=WORLD_SIZE, rank=RANK)

        for i in range(len(shard_dataset)):
            # Check if image is correct.
            ind = i * WORLD_SIZE + RANK
            self.assertTrue(
                torch.all(
                    shard_dataset[i]["input"] == test_dataset[ind]["input"]
                ).item()
                == 1
            )
            # Check if target is correct
            self.assertTrue(
                torch.all(
                    shard_dataset[i]["target"] == test_dataset[ind]["target"]
                ).item()
                == 1
            )
            self.assertTrue(shard_dataset[i]["is_dummy_sample"] == torch.tensor(0))

    def test_final_shard_dataset_index(self):
        """
        Test that for final shard, the indices map to correct indices from the
        main dataset and is_dummy_sample is as expected.
        """
        test_dataset, _ = create_test_dataset(tensor_size=(14, 3, 224, 224))
        WORLD_SIZE = 4
        RANK = 3
        shard_dataset = ShardDataset(test_dataset, world_size=WORLD_SIZE, rank=RANK)

        # Check that for both rank 3 and 4, first 3 samples are real and
        # last one is dummy.
        for i in range(3):
            # Check if image is correct.
            ind = i * WORLD_SIZE + RANK
            self.assertTrue(
                torch.all(
                    shard_dataset[i]["input"] == test_dataset[ind]["input"]
                ).item()
                == 1
            )
            # Check if target is correct
            self.assertTrue(
                torch.all(
                    shard_dataset[i]["target"] == test_dataset[ind]["target"]
                ).item()
                == 1
            )
            self.assertTrue(shard_dataset[i]["is_dummy_sample"] == torch.tensor(0))

        # Check that last two samples are dummy samples.
        self.assertTrue(shard_dataset[3]["is_dummy_sample"] == torch.tensor(1))
