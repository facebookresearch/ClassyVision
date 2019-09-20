#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
import sys
import unittest
from test.generic.merge_dataset import MergeDataset
from test.generic.utils import (
    compare_batches,
    compare_datasets,
    create_test_dataset,
    make_torch_deterministic,
)

import torch
from classy_vision.dataset.core import (
    BatchDataset,
    ResampleDataset,
    ShuffleDataset,
    TransformDataset,
)
from classy_vision.dataset.core.async_dataset_iterator import AsyncDatasetIterator
from classy_vision.dataset.core.dataset_iterator import DatasetIterator


# tensor sizes for which to run tests:
TENSOR_SIZES = []
for height in [1, 5]:
    for width in [1, 5]:
        TENSOR_SIZES.append((height, width))
for _ in range(2):  # testing 3D and 4D tensors
    TENSOR_SIZES.extend(
        [tuple([dim_size] + list(size)) for size in TENSOR_SIZES for dim_size in [1, 5]]
    )

# transforms for tests:
ALL_TRANSFORMS = [
    lambda x: {"input": x["input"] + 1.0, "target": x["target"]},
    lambda x: {"input": x["input"] * 2.0, "target": x["target"]},
    lambda x: {"input": x["input"] * 2.0 + 1.0, "target": x["target"]},
]


class TestDatasetDSL(unittest.TestCase):
    """Tests dataset DSL functions."""

    def _compare_merge_datasets(self, dataset1, dataset2):
        """Compares two merged dataset objects."""
        self.assertEqual(len(dataset1), len(dataset2))
        self.assertEqual(len(dataset1.datasets), len(dataset2.datasets))
        self.assertEqual(type(dataset1.datasets), type(dataset2.datasets))
        generator = (
            range(len(dataset1.datasets))
            if isinstance(dataset1.datasets, list)
            else dataset1.keys()
        )
        for key in generator:
            for idx in range(len(dataset1)):
                compare_batches(
                    self, dataset1.datasets[key][idx], dataset2.datasets[key][idx]
                )

    def test_regular(self):
        """Tests the DSL on regular Dataset objects."""
        for size in TENSOR_SIZES:

            # create dataset:
            dataset1, _ = create_test_dataset(size)
            dataset2 = copy.deepcopy(dataset1)
            compare_datasets(self, dataset1, dataset2)

            # test transforms:
            for transform in ALL_TRANSFORMS:
                dataset1 = dataset1.transform(transform)
                dataset2 = TransformDataset(dataset2, transform)
                compare_datasets(self, dataset1, dataset2)

            # test resampling:
            resample = torch.randperm(len(dataset1)).tolist()
            dataset1 = dataset1.resample(resample)
            dataset2 = ResampleDataset(dataset2, resample)
            compare_datasets(self, dataset1, dataset2)

            # test shuffling:
            make_torch_deterministic()
            dataset1 = dataset1.shuffle()
            make_torch_deterministic()
            dataset2 = ShuffleDataset(dataset2)
            compare_datasets(self, dataset1, dataset2)

            # test batching:
            for batchsize_per_replica in range(1, len(dataset1) - 1):
                for skip_last in [True, False]:
                    batch_dataset1 = dataset1.batch(
                        batchsize_per_replica, skip_last=skip_last
                    )
                    batch_dataset2 = BatchDataset(
                        dataset2, batchsize_per_replica, skip_last=skip_last
                    )
                    compare_datasets(self, batch_dataset1, batch_dataset2)

            # Verify iterator types, does not verify iteration, this
            # is done in separate test set
            it = dataset1.batch(batchsize_per_replica=1).iterator()
            self.assertTrue(isinstance(it, DatasetIterator))

            it = dataset1.batch(batchsize_per_replica=1).iterator(num_workers=2)
            self.assertTrue(isinstance(it, AsyncDatasetIterator))
            self.assertEqual(it.num_workers, 2)

    def test_recursive_state(self):
        """Recursive state tests

        We modify the get / set state functions on some of the dataset
        objects. This requires recursively getting / setting state
        through the wrappers. This test adds a few layers of recursion
        and then uses deepcopy to verify we get the same dataset.

        Note, the goal of this test is not to verify every possible
        combination of wrappers, it's just to verify the recursion
        works in a few different scenarios. Users adding state to
        their datasets are responsible for testing how it behaves with
        the larger framework.

        """
        # create dataset:
        for size in TENSOR_SIZES:
            base_dataset, _ = create_test_dataset(size)
            for transform in ALL_TRANSFORMS:
                for batchsize_per_replica in range(1, len(base_dataset) - 1):
                    # Test shuffle - shuffle
                    dataset1 = base_dataset.transform(transform)
                    dataset2 = base_dataset.transform(transform)
                    dataset1 = dataset1.shuffle()
                    dataset2 = dataset2.shuffle()
                    dataset1 = dataset1.batch(batchsize_per_replica)
                    dataset2 = dataset2.batch(batchsize_per_replica)
                    dataset1.do_shuffle(epoch_num=0)
                    dataset2.do_shuffle(epoch_num=2)

                    state1 = dataset1.get_classy_state()
                    dataset2.set_classy_state(state1)
                    state2 = dataset2.get_classy_state()
                    self.assertTrue(state1 == state2)

                    compare_datasets(self, dataset1, dataset2)

                    # Test merge - shuffle
                    dataset1 = base_dataset.transform(transform)
                    dataset2 = base_dataset.transform(transform)
                    dataset1 = dataset1.shuffle()
                    dataset2 = dataset2.shuffle()
                    dataset1 = dataset1.batch(batchsize_per_replica)
                    dataset2 = dataset2.batch(batchsize_per_replica)
                    dataset3 = MergeDataset(
                        [copy.deepcopy(dataset1), copy.deepcopy(dataset2)]
                    )
                    dataset1.do_shuffle(epoch_num=1)
                    dataset2.do_shuffle(epoch_num=5)
                    dataset4 = MergeDataset([dataset1, dataset2])
                    state4 = dataset4.get_classy_state()
                    dataset3.set_classy_state(state4)
                    state3 = dataset3.get_classy_state()
                    self.assertTrue(state3 == state4)

                    self._compare_merge_datasets(dataset3, dataset4)


# run all the tests:
if __name__ == "__main__":
    unittest.main(argv=sys.argv[0])
