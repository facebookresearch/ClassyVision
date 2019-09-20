#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.utils import (
    MockErrorDataset,
    compare_batchlist_and_dataset_with_skips,
    create_test_dataset,
    make_torch_deterministic,
    recursive_unpack,
)

import torch
from classy_vision.dataset.core.async_dataset_iterator import AsyncDatasetIterator
from classy_vision.dataset.core.base_async_dataset_iterator import (
    NUM_SAMPLES_TO_PREFETCH,
)


MP_START_METHODS = ["spawn", "fork", "forkserver"]


class TestAsyncIterators(unittest.TestCase):
    """
    Tests async iterators
    """

    def _compare_batchlist_and_dataset(self, batch_list, dataset, skip_indices=None):
        compare_batchlist_and_dataset_with_skips(
            self, batch_list, dataset, skip_indices
        )

    def _get_test_dataset(self, length=10):
        """Returns a mock dataset"""
        dataset, _ = create_test_dataset((length, 2, 2))
        return dataset

    def setUp(self):
        make_torch_deterministic()

    def test_dataset_iterator(self):
        for mp_start_method in MP_START_METHODS:
            print("Start method {}".format(mp_start_method))
            for num_workers in [1, 2, 3]:
                dataset = self._get_test_dataset().batch(batchsize_per_replica=2)
                it = AsyncDatasetIterator(
                    dataset, num_workers=num_workers, mp_start_method=mp_start_method
                )
                batch_list = [batch for batch in it]
                self._compare_batchlist_and_dataset(batch_list, dataset)

            # Verify iterator works with num_workers > num_batches
            dataset = self._get_test_dataset().batch(batchsize_per_replica=5)
            it = AsyncDatasetIterator(
                dataset, num_workers=3, mp_start_method=mp_start_method
            )
            batch_list = [batch for batch in it]
            self._compare_batchlist_and_dataset(batch_list, dataset)
            print("Start method {} test complete".format(mp_start_method))

    def test_backfill_dataset_iterator(self):
        for mp_start_method in MP_START_METHODS:
            print("Start method {}".format(mp_start_method))
            dataset = self._get_test_dataset().batch(batchsize_per_replica=2)
            it = AsyncDatasetIterator(
                dataset,
                num_workers=1,
                backfill_batches=True,
                mp_start_method=mp_start_method,
            )
            batch_list = [batch for batch in it]
            self._compare_batchlist_and_dataset(batch_list, dataset)

            # Backfill test, num_workers = 1 so we maintain order
            dataset = self._get_test_dataset()
            error_dataset = MockErrorDataset(dataset.batch(batchsize_per_replica=2))
            # Resize batch 3 to size 1, simulates error on sample 3 * 2 + 1 = 7
            error_dataset.rebatch_map[3] = 1
            # Resize batch 1 to size 0, simulates error on samples 2 and 3
            error_dataset.rebatch_map[1] = 0
            it = AsyncDatasetIterator(
                error_dataset,
                num_workers=1,
                backfill_batches=True,
                mp_start_method=mp_start_method,
            )
            batch_list = [recursive_unpack(batch) for batch in it]
            batch_list = [item for sublist in batch_list for item in sublist]
            # Skip sample 2,3,7
            self._compare_batchlist_and_dataset(
                batch_list, dataset, skip_indices=[2, 3, 7]
            )

            # Backfill test, num_workers = 2, so we can't guarantee
            # order. Will verify number / size of batches and each sample
            # was in dataset. Same errors as previous test.
            dataset = self._get_test_dataset()
            error_dataset = MockErrorDataset(dataset.batch(batchsize_per_replica=2))
            # Resize batch 3 to size 1, simulates error on sample 3 * 2 + 1 = 7
            error_dataset.rebatch_map[3] = 1
            # Resize batch 1 to size 0, simulates error on samples 2 and 3
            error_dataset.rebatch_map[1] = 0
            it = AsyncDatasetIterator(
                error_dataset,
                num_workers=2,
                backfill_batches=True,
                mp_start_method=mp_start_method,
            )
            batch_list = [batch for batch in it]
            self.assertEqual(len(batch_list), 4)
            for idx, batch in enumerate(batch_list):
                if idx != len(batch_list) - 1:
                    self.assertEqual(batch["input"].size()[0], 2)
                else:
                    self.assertEqual(batch["input"].size()[0], 1)

            # Check that each sample is in dataset. This is N^2, where N
            # is determined in _get_test_dataset()
            batch_list = [recursive_unpack(batch) for batch in batch_list]
            batch_list = [item for sublist in batch_list for item in sublist]
            for sample_i in batch_list:
                found_batch = False
                for sample_j in dataset:
                    if torch.allclose(
                        sample_i["input"], sample_j["input"]
                    ) and torch.allclose(sample_i["target"], sample_j["target"]):
                        found_batch = True
                        break

                self.assertTrue(found_batch)

            # Verify that this works if num_workers > num_batches. Will
            # use batchsize_per_replica of 5, which should result in 2 batches
            dataset = self._get_test_dataset()
            batched_dataset = dataset.batch(batchsize_per_replica=5)
            it = AsyncDatasetIterator(
                batched_dataset,
                num_workers=3,
                backfill_batches=True,
                mp_start_method=mp_start_method,
            )
            batch_list = [batch for batch in it]
            self.assertEqual(len(batch_list), 2)
            for batch in batch_list:
                self.assertEqual(batch["input"].size()[0], 5)

            # Check that each sample is in dataset. This is N^2, where N
            # is determined in _get_test_dataset()
            batch_list = [recursive_unpack(batch) for batch in batch_list]
            batch_list = [item for sublist in batch_list for item in sublist]
            for sample_i in batch_list:
                found_batch = False
                for sample_j in dataset:
                    if torch.allclose(
                        sample_i["input"], sample_j["input"]
                    ) and torch.allclose(sample_i["target"], sample_j["target"]):
                        found_batch = True
                        break

                self.assertTrue(found_batch)
            print("Start method {} test complete".format(mp_start_method))

    def test_backfill_dataset_iterator_timeout_failures(self):
        for mp_start_method in MP_START_METHODS:
            print("Start method {} test begun".format(mp_start_method))
            # Check that dataloader works when NUM_SAMPLES_TO_PREFETCH
            # batches together are too small (1 sample instead of
            # 2). Note, one potential failure mode of this test is
            # timeouts which is why I separated it from the others
            dataset = self._get_test_dataset(length=3 * NUM_SAMPLES_TO_PREFETCH)
            error_dataset = MockErrorDataset(dataset.batch(batchsize_per_replica=2))
            error_dataset.rebatch_map[0] = 1
            skip_indices = [1]
            for i in range(1, NUM_SAMPLES_TO_PREFETCH):
                error_dataset.rebatch_map[i] = 0
                skip_indices.append(2 * i)
                skip_indices.append(2 * i + 1)

            it = AsyncDatasetIterator(
                error_dataset,
                num_workers=1,
                backfill_batches=True,
                mp_start_method=mp_start_method,
            )
            batch_list = [recursive_unpack(batch) for batch in it]
            batch_list = [item for sublist in batch_list for item in sublist]
            self._compare_batchlist_and_dataset(
                batch_list, dataset, skip_indices=skip_indices
            )
            print("Start method {} test complete".format(mp_start_method))
