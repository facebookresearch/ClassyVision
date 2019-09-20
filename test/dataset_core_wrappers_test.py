#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
import random
import re
import sys
import unittest
from test.generic.utils import (
    create_test_data,
    create_test_dataset,
    create_test_targets,
)

import torch
from classy_vision.dataset.core import (
    BatchDataset,
    Dataset,
    ListDataset,
    ResampleDataset,
    ShuffleDataset,
    TransformDataset,
    WrapDataset,
)


# tensor sizes for which to run tests:
TENSOR_SIZES = []
for height in [1, 2, 5, 10]:
    for width in [1, 2, 5, 10]:
        TENSOR_SIZES.append((height, width))
for _ in range(2):  # testing 3D and 4D tensors
    TENSOR_SIZES.extend(
        [
            tuple([dim_size] + list(size))
            for size in TENSOR_SIZES
            for dim_size in [1, 2, 5, 10]
        ]
    )


# transforms for tests:
def transform(x, mult, add):
    input_sample = x["input"]
    target_sample = x["target"]

    return {"input": input_sample * mult + add, "target": target_sample}


ALL_TRANSFORMS = [
    functools.partial(transform, mult=1.0, add=1.0),
    functools.partial(transform, mult=2.0, add=0.0),
    functools.partial(transform, mult=2.0, add=1.0),
]


class TestDataset(unittest.TestCase):
    """Tests torchnet dataset implementations."""

    def _compare_dataset_with_values(self, dataset, values, resample=None):
        """
        Compares a dataset to a dict -> list of dataset tensors. The
        optional resample list indicates how values ought to be
        ordered in dataset.
        """
        if resample is None:
            resample = [idx for idx in range(len(dataset))]
        self.assertEqual(len(dataset), len(values["input"]))
        for idx in torch.randperm(len(dataset)).tolist():
            self.assertTrue(isinstance(dataset[idx], dict))
            self.assertEqual(len(dataset[idx]), len(values))
            for key in values.keys():
                sample = dataset[idx][key]
                # PyTorch tensor dataset wraps data in tuples
                if isinstance(sample, (list, tuple)):
                    assert len(sample) == 1, "Expected sample to have length 1"
                    sample = sample[0]
                self.assertTrue(torch.allclose(sample, values[key][resample[idx]]))

    def _compare_datasets(self, dataset1, dataset2):
        self.assertEqual(len(dataset1), len(dataset2))
        for idx in range(len(dataset1)):
            self.assertTrue(
                torch.allclose(dataset1[idx]["input"], dataset2[idx]["input"])
            )
            self.assertTrue(
                torch.allclose(dataset1[idx]["target"], dataset2[idx]["target"])
            )

    def _verify_classy_state(self, dataset, exp_state=None):
        """
        This only needs to be called for terminal wrappers (e.g. list
        dataset or wrap dataset. Verifies that call does not break, in
        these cases state returned should be the default dataset type
        and the set function should just return self.
        """
        state = dataset.get_classy_state()
        if exp_state is None:
            exp_state = {"state": {"dataset_type": type(dataset)}}
        self.assertEqual(state, exp_state)
        dataset = dataset.set_classy_state(state)
        self.assertTrue(isinstance(dataset, Dataset))

    def test_batch_dataset(self):
        for size in TENSOR_SIZES:
            dataset, values = create_test_dataset(size)
            for batchsize_per_replica in range(1, len(dataset) + 1):
                for skip_last in [True, False]:

                    # check dataset size:
                    batch_dataset = BatchDataset(
                        dataset, batchsize_per_replica, skip_last=skip_last
                    )
                    dataset_len = float(len(dataset)) / float(batchsize_per_replica)
                    dataset_len = int(
                        math.floor(dataset_len) if skip_last else math.ceil(dataset_len)
                    )
                    self.assertEqual(len(batch_dataset), dataset_len)

                    # check all batches:
                    sample_idx = 0
                    for batch_idx in range(len(batch_dataset)):

                        # check size of batch:
                        batch = batch_dataset[batch_idx]
                        if not skip_last and batch_idx == len(batch_dataset) - 1:
                            final_batchsize = len(dataset) % batchsize_per_replica
                            if final_batchsize == 0:
                                final_batchsize = batchsize_per_replica
                            self.assertEqual(batch["input"].size(0), final_batchsize)
                        else:
                            self.assertEqual(
                                batch["input"].size(0), batchsize_per_replica
                            )

                        # check values in batch:
                        for idx in range(batch["input"].size(0)):
                            self.assertTrue(
                                torch.allclose(
                                    batch["input"][idx], values["input"][sample_idx]
                                )
                            )
                            self.assertEqual(
                                batch["target"][idx].item(),
                                values["target"][sample_idx].item(),
                            )
                            sample_idx += 1

    def test_list_dataset(self):
        for size in TENSOR_SIZES:
            dataset, values = create_test_dataset(size)
            metadata = [{"target": values["target"][ind]} for ind in range(size[0])]
            list_dataset = ListDataset(values["input"], metadata, lambda x: x)
            self._compare_dataset_with_values(list_dataset, values)

            target = [values["target"][ind] for ind in range(size[0])]
            list_dataset_2 = ListDataset(values["input"], target, lambda x: x)
            self._compare_dataset_with_values(list_dataset_2, values)
            self._verify_classy_state(list_dataset)

    def test_mmap_dataset(self):
        pass  # TODO

    def test_resample_dataset(self):
        for size in TENSOR_SIZES:
            dataset, values = create_test_dataset(size)
            resample = torch.randperm(len(dataset)).tolist()
            resample_dataset = ResampleDataset(dataset, resample)
            self._compare_dataset_with_values(
                resample_dataset, values, resample=resample
            )

    def test_shuffle_dataset(self):
        for size in TENSOR_SIZES:
            dataset, values = create_test_dataset(size)
            shuffle_dataset = ShuffleDataset(dataset)
            self._compare_dataset_with_values(
                shuffle_dataset, values, resample=shuffle_dataset._resample
            )
            shuffle_dataset.do_shuffle(epoch_num=0)
            self._compare_dataset_with_values(
                shuffle_dataset, values, resample=shuffle_dataset._resample
            )

            # Test get / set state
            new_shuffle_dataset = ShuffleDataset(dataset)
            shuffle_dataset.do_shuffle(epoch_num=1)
            state = shuffle_dataset.get_classy_state()
            new_shuffle_dataset.set_classy_state(state)
            self._compare_datasets(shuffle_dataset, new_shuffle_dataset)

            # Test multiple levels of shuffling
            shuffle_dataset = ShuffleDataset(shuffle_dataset)
            shuffle_dataset.do_shuffle(epoch_num=2)
            new_shuffle_dataset = ShuffleDataset(ShuffleDataset(dataset))
            state = shuffle_dataset.get_classy_state()
            new_shuffle_dataset.set_classy_state(state)
            self._compare_datasets(shuffle_dataset, new_shuffle_dataset)
            self._compare_datasets(shuffle_dataset.dataset, new_shuffle_dataset.dataset)

            # Test that we get the same shuffle when provided with the same
            # seed and epoch_num
            shuffle_dataset = ShuffleDataset(dataset, seed=1)
            shuffle_dataset_2 = ShuffleDataset(dataset, seed=1)
            shuffle_dataset.do_shuffle(epoch_num=0)
            shuffle_dataset_2.do_shuffle(epoch_num=0)
            self._compare_datasets(shuffle_dataset.dataset, shuffle_dataset_2.dataset)

    def test_transform_dataset(self):
        for size in TENSOR_SIZES:
            for transform in ALL_TRANSFORMS:
                dataset, values = create_test_dataset(size)
                transform_dataset = TransformDataset(dataset, transform)
                self._compare_dataset_with_values(transform_dataset, transform(values))

    def test_wrap_dataset(self):
        for size in TENSOR_SIZES:
            _, values = create_test_data(size)
            dataset = torch.utils.data.TensorDataset(values)
            wrap_dataset = WrapDataset(dataset)
            self.assertEqual(len(wrap_dataset), len(dataset))
            for idx in torch.randperm(len(dataset)).tolist():
                self.assertTrue(
                    torch.allclose(wrap_dataset[idx]["input"][0], dataset[idx][0])
                )
            self._verify_classy_state(wrap_dataset)


# run all the tests:
if __name__ == "__main__":
    unittest.main(argv=sys.argv[0])
