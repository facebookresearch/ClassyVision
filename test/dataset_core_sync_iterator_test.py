#!/usr/bin/env python3

import unittest
from test.generic.utils import (
    MockErrorDataset,
    compare_batches,
    compare_batchlist_and_dataset_with_skips,
    create_test_dataset,
    make_torch_deterministic,
    recursive_unpack,
)

import torch
from classy_vision.dataset.core.backfill_async_dataset_iterator import (
    backfill_batch,
    recursive_batchsize_per_replica,
)
from classy_vision.dataset.core.dataset_iterator import DatasetIterator


class TestBackfillHelper(unittest.TestCase):
    """
    Tests Backfill helper function
    """

    def _compare_batches(self, batch1, batch2):
        """Compares two batches."""
        compare_batches(self, batch1, batch2)

    def setUp(self):
        make_torch_deterministic()

    def test_backfill_list_batch(self):
        batchsize_per_replica = 10
        # None and too small batch
        batch = [torch.tensor([]).float(), torch.tensor([]).int()]
        next_batch = [torch.ones([5, 5, 5]).float(), torch.ones([5]).int()]
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self.assertEqual(final_batch, None)
        self._compare_batches(unfinished_batch, next_batch)

        # Combined batches are not big enough
        batch = [torch.ones([4, 5, 5]).float(), torch.ones([4]).int()]
        next_batch = [torch.ones([4, 5, 5]).float(), torch.ones([4]).int()]
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self.assertEqual(final_batch, None)
        self._compare_batches(
            unfinished_batch,
            [
                torch.ones([8, 5, 5], dtype=torch.float),
                torch.ones([8], dtype=torch.int),
            ],
        )

        # Basic backfilling
        batch = [torch.ones([4, 5, 5]).float(), torch.ones([4]).int()]
        next_batch = [torch.ones([6, 5, 5]).float(), torch.ones([6]).int()]
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self._compare_batches(
            final_batch, [torch.ones([10, 5, 5]).float(), torch.ones([10]).int()]
        )
        self._compare_batches(unfinished_batch, [None, None])

        batch = [torch.tensor([]).float(), torch.tensor([]).int()]
        next_batch = [torch.ones([10, 5, 5]).float(), torch.ones([10]).int()]
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self._compare_batches(
            final_batch, [torch.ones([10, 5, 5]).float(), torch.ones([10]).int()]
        )
        self._compare_batches(unfinished_batch, [None, None])

        # Combined batches are more than big enough
        batch = [torch.ones([4, 5, 5]).float(), torch.ones([4]).int()]
        next_batch = [torch.ones([8, 5, 5]).float(), torch.ones([8]).int()]
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self._compare_batches(
            final_batch, [torch.ones([10, 5, 5]).float(), torch.ones([10]).int()]
        )
        self._compare_batches(
            unfinished_batch,
            [
                torch.ones([2, 5, 5], dtype=torch.float),
                torch.ones([2], dtype=torch.int),
            ],
        )

        # Batch is bigger than batchsize_per_replica
        batch = [torch.ones([11, 5, 5]).float(), torch.ones([4]).int()]
        next_batch = [torch.ones([8, 5, 5]).float(), torch.ones([8]).int()]

        with self.assertRaises(AssertionError):
            final_batch, unfinished_batch = backfill_batch(
                batch, next_batch, batchsize_per_replica
            )

        # Next batch is bigger than batchsize_per_replica
        batch = [torch.ones([4, 5, 5]).float(), torch.ones([4]).int()]
        next_batch = [torch.ones([11, 5, 5]).float(), torch.ones([8]).int()]

        with self.assertRaises(AssertionError):
            final_batch, unfinished_batch = backfill_batch(
                batch, next_batch, batchsize_per_replica
            )

    def test_backfill_unacceptable_type(self):
        batchsize_per_replica = 10
        batch = ["string", 123, 1.0]
        next_batch = ["string", 123, 1.0]
        with self.assertRaises(TypeError):
            final_batch, unfinished_batch = backfill_batch(
                batch, next_batch, batchsize_per_replica
            )

        batch = {"input": "string", "target": 123, "test_weight": 1.0}
        next_batch = {"input": "string", "target": 123, "test_weight": 1.0}
        with self.assertRaises(TypeError):
            final_batch, unfinished_batch = backfill_batch(
                batch, next_batch, batchsize_per_replica
            )

        batch = "string"
        next_batch = "string"
        with self.assertRaises(TypeError):
            final_batch, unfinished_batch = backfill_batch(
                batch, next_batch, batchsize_per_replica
            )

    def test_backfill_tuple_batch(self):
        batchsize_per_replica = 10
        # None and too small batch
        batch = (torch.tensor([]).float(), torch.tensor([]).int())
        next_batch = (torch.ones([5, 5, 5]).float(), torch.ones([5]).int())
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self.assertEqual(final_batch, None)
        self._compare_batches(unfinished_batch, next_batch)

        # Combined batches are not big enough
        batch = (torch.ones([4, 5, 5]).float(), torch.ones([4]).int())
        next_batch = (torch.ones([4, 5, 5]).float(), torch.ones([4]).int())
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self.assertEqual(final_batch, None)
        self._compare_batches(
            unfinished_batch, (torch.ones([8, 5, 5]).float(), torch.ones([8]).int())
        )

        # Basic backfilling
        batch = (torch.ones([4, 5, 5]).float(), torch.ones([4]).int())
        next_batch = (torch.ones([6, 5, 5]).float(), torch.ones([6]).int())
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self._compare_batches(
            final_batch, (torch.ones([10, 5, 5]).float(), torch.ones([10]).int())
        )
        self._compare_batches(unfinished_batch, (None, None))

        batch = (torch.tensor([]).float(), torch.tensor([]).int())
        next_batch = (torch.ones([10, 5, 5]).float(), torch.ones([10]).int())
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self._compare_batches(
            final_batch, (torch.ones([10, 5, 5]).float(), torch.ones([10]).int())
        )
        self._compare_batches(unfinished_batch, (None, None))

        # Combined batches are more than big enough
        batch = (torch.ones([4, 5, 5]).float(), torch.ones([4]).int())
        next_batch = (torch.ones([8, 5, 5]).float(), torch.ones([8]).int())
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self._compare_batches(
            final_batch, (torch.ones([10, 5, 5]).float(), torch.ones([10]).int())
        )
        self._compare_batches(
            unfinished_batch,
            (
                torch.ones([2, 5, 5], dtype=torch.float),
                torch.ones([2], dtype=torch.int),
            ),
        )

        # Batch is bigger than batchsize_per_replica
        batch = (torch.ones([11, 5, 5]).float(), torch.ones([4]).int())
        next_batch = (torch.ones([8, 5, 5]).float(), torch.ones([8]).int())

        with self.assertRaises(AssertionError):
            final_batch, unfinished_batch = backfill_batch(
                batch, next_batch, batchsize_per_replica
            )

        # Next batch is bigger than batchsize_per_replica
        batch = (torch.ones([4, 5, 5]).float(), torch.ones([4]).int())
        next_batch = (torch.ones([11, 5, 5]).float(), torch.ones([8]).int())

        with self.assertRaises(AssertionError):
            final_batch, unfinished_batch = backfill_batch(
                batch, next_batch, batchsize_per_replica
            )

    def test_backfill_dict_batch(self):
        batchsize_per_replica = 10
        # None and too small batch
        batch = {"input": torch.tensor([]).float(), "target": torch.tensor([]).int()}
        next_batch = {
            "input": torch.ones([5, 5, 5]).float(),
            "target": torch.ones([5]).int(),
        }
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self.assertEqual(final_batch, None)
        self._compare_batches(unfinished_batch, next_batch)

        # Combined batches are not big enough
        batch = {
            "input": torch.ones([4, 5, 5]).float(),
            "target": torch.ones([4]).int(),
        }
        next_batch = {
            "input": torch.ones([4, 5, 5]).float(),
            "target": torch.ones([4]).int(),
        }
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self.assertEqual(final_batch, None)
        self._compare_batches(
            unfinished_batch,
            {
                "input": torch.ones([8, 5, 5], dtype=torch.float),
                "target": torch.ones([8], dtype=torch.int),
            },
        )

        # Basic backfilling
        batch = {
            "input": torch.ones([4, 5, 5]).float(),
            "target": torch.ones([4]).int(),
        }
        next_batch = {
            "input": torch.ones([6, 5, 5]).float(),
            "target": torch.ones([6]).int(),
        }
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self._compare_batches(
            final_batch,
            {"input": torch.ones([10, 5, 5]).float(), "target": torch.ones([10]).int()},
        )
        self._compare_batches(unfinished_batch, {"input": None, "target": None})

        batch = {"input": torch.tensor([]).float(), "target": torch.tensor([]).int()}
        next_batch = {
            "input": torch.ones([10, 5, 5]).float(),
            "target": torch.ones([10]).int(),
        }
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self._compare_batches(
            final_batch,
            {"input": torch.ones([10, 5, 5]).float(), "target": torch.ones([10]).int()},
        )
        self._compare_batches(unfinished_batch, {"input": None, "target": None})

        # Combined batches are more than big enough
        batch = {
            "input": torch.ones([4, 5, 5]).float(),
            "target": torch.ones([4]).int(),
        }
        next_batch = {
            "input": torch.ones([8, 5, 5]).float(),
            "target": torch.ones([8]).int(),
        }
        final_batch, unfinished_batch = backfill_batch(
            batch, next_batch, batchsize_per_replica
        )

        self._compare_batches(
            final_batch,
            {"input": torch.ones([10, 5, 5]).float(), "target": torch.ones([10]).int()},
        )
        self._compare_batches(
            unfinished_batch,
            {
                "input": torch.ones([2, 5, 5], dtype=torch.float),
                "target": torch.ones([2], dtype=torch.int),
            },
        )

        # Batch is bigger than batchsize_per_replica
        batch = {
            "input": torch.ones([11, 5, 5]).float(),
            "target": torch.ones([4]).int(),
        }
        next_batch = {
            "input": torch.ones([8, 5, 5]).float(),
            "target": torch.ones([8]).int(),
        }

        with self.assertRaises(AssertionError):
            final_batch, unfinished_batch = backfill_batch(
                batch, next_batch, batchsize_per_replica
            )

        # Next batch is bigger than batchsize_per_replica
        batch = {
            "input": torch.ones([4, 5, 5]).float(),
            "target": torch.ones([4]).int(),
        }
        next_batch = {
            "input": torch.ones([11, 5, 5]).float(),
            "target": torch.ones([8]).int(),
        }

        with self.assertRaises(AssertionError):
            final_batch, unfinished_batch = backfill_batch(
                batch, next_batch, batchsize_per_replica
            )

    def test_recursive_batchsize_per_replica(self):
        self.assertEqual(0, recursive_batchsize_per_replica(None))
        for batchsize_per_replica in [1, 5, 10]:
            input = torch.randn([batchsize_per_replica, 10, 10])
            target = torch.randint(0, 10, (batchsize_per_replica,))
            # tuple cases
            self.assertEqual(
                batchsize_per_replica, recursive_batchsize_per_replica((input, target))
            )
            # list cases
            self.assertEqual(
                batchsize_per_replica, recursive_batchsize_per_replica([input, target])
            )
            # dict cases
            self.assertEqual(
                batchsize_per_replica,
                recursive_batchsize_per_replica({"input": input, "target": target}),
            )
            # Mixed cases
            complex_sample = {
                "test_field1": {
                    "test_field2": [(input.clone(), input.clone())],
                    "test_field3": input.clone(),
                },
                "test_field4": (target.clone(), target.clone()),
            }
            self.assertEqual(
                batchsize_per_replica, recursive_batchsize_per_replica(complex_sample)
            )
            # Failure (mismatched batchsize_per_replicas)
            complex_sample = {
                "test_field1": input.clone(),
                "test_field2": [torch.randn([20, 10, 10])],
                "test_field3": target.clone(),
            }
            with self.assertRaises(AssertionError):
                recursive_batchsize_per_replica(complex_sample)
            # Failure (bad types)
            complex_sample = {"test_field1": [(23,)], "test_field2": input.clone()}
            with self.assertRaises(TypeError):
                recursive_batchsize_per_replica(complex_sample)


class TestSyncIterators(unittest.TestCase):
    """
    Tests sync iterators
    """

    def _compare_batchlist_and_dataset(self, batch_list, dataset, skip_indices=None):
        """Compares a batch list and a dataset, skips allowed"""
        compare_batchlist_and_dataset_with_skips(
            self, batch_list, dataset, skip_indices
        )

    def _get_test_dataset(self):
        """Returns a mock dataset"""
        dataset, _ = create_test_dataset((10, 2, 2))
        return dataset

    def setUp(self):
        make_torch_deterministic()

    def test_dataset_iterator(self):
        dataset = self._get_test_dataset()
        it = DatasetIterator(dataset)
        batch_list = [batch for batch in it]
        self._compare_batchlist_and_dataset(batch_list, dataset)

    def test_backfill_dataset_iterator(self):
        dataset = self._get_test_dataset().batch(batchsize_per_replica=2)
        it = DatasetIterator(dataset, backfill_batches=True)
        # Note, this test currently doesn't test backfilling a missing batch
        batch_list = [batch for batch in it]
        self._compare_batchlist_and_dataset(batch_list, dataset)

        # Backfill test with errors
        dataset = self._get_test_dataset()
        error_dataset = MockErrorDataset(dataset.batch(batchsize_per_replica=2))
        # Resize batch 3 to size 1, simulates error on sample 3 * 2 + 1 = 7
        error_dataset.rebatch_map[3] = 1
        # Resize batch 1 to size 0, simulates error on samples 2 and 3
        error_dataset.rebatch_map[1] = 0
        it = DatasetIterator(error_dataset, backfill_batches=True)
        batch_list = [recursive_unpack(batch) for batch in it]
        batch_list = [item for sublist in batch_list for item in sublist]
        # Skip sample 2,3,7
        self._compare_batchlist_and_dataset(batch_list, dataset, skip_indices=[2, 3, 7])

        # Ensure backfilling works for different batchsize_per_replica
        dataset = self._get_test_dataset()
        error_dataset = MockErrorDataset(dataset.batch(batchsize_per_replica=3))
        # Resize batch 1 to size 2, simulates error on sample 2 * 3 - 1 = 5
        error_dataset.rebatch_map[1] = 2
        # Resize batch 0 to size 0, simulates error on samples 0, 1, 2
        error_dataset.rebatch_map[0] = 0
        it = DatasetIterator(error_dataset, backfill_batches=True)
        batch_list = [recursive_unpack(batch) for batch in it]
        batch_list = [item for sublist in batch_list for item in sublist]
        # Skip sample 0, 1, 2, 5
        self._compare_batchlist_and_dataset(
            batch_list, dataset, skip_indices=[0, 1, 2, 5]
        )
