#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock as mock
from test.generic.utils import compare_batches, compare_samples

import classy_vision.dataset.classy_dataset as classy_dataset
import torch
from classy_vision.dataset import build_dataset, get_available_splits, register_dataset
from classy_vision.dataset.core import ListDataset
from torch.utils.data import DataLoader


DUMMY_SAMPLES_1 = [
    {"input": torch.tensor([[[0, 1], [2, 3]]]), "target": torch.tensor([[0]])}
]


DUMMY_SAMPLES_2 = [
    {"input": torch.tensor([[[0, 1], [2, 3]]]), "target": torch.tensor([[0]])},
    {"input": torch.tensor([[[4, 5], [6, 7]]]), "target": torch.tensor([[1]])},
]

BATCHED_DUMMY_SAMPLES_2 = [
    {
        "input": torch.tensor([[[[0, 1], [2, 3]]], [[[4, 5], [6, 7]]]]),
        "target": torch.tensor([[[0]], [[1]]]),
    }
]

DUMMY_CONFIG = {"name": "test_dataset", "dummy0": 0, "dummy1": 1}

OTHER_DUMMY_CONFIG = {"name": "other_test_dataset", "dummy0": 0, "dummy1": 1}


def mock_get_world_size():
    return 2


def mock_get_rank():
    return 1


@register_dataset("test_dataset")
class TestDataset(classy_dataset.ClassyDataset):
    """Test dataset for validating registry functions"""

    def __init__(
        self, samples, batchsize_per_replica=1, num_samples=None, transform=None
    ):
        super().__init__(
            split=None,
            batchsize_per_replica=batchsize_per_replica,
            shuffle=False,
            transform=transform,
            num_samples=len(samples) if num_samples is None else num_samples,
        )
        input_tensors = [sample["input"] for sample in samples]
        target_tensors = [sample["target"] for sample in samples]
        self.dataset = ListDataset(input_tensors, target_tensors, loader=lambda x: x)

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        return cls(*args, **kwargs)


@register_dataset("other_test_dataset")
class OtherTestDataset(classy_dataset.ClassyDataset):
    """
    Test dataset for validating registry functions that has a different
    type than TestDataset
    """

    @classmethod
    def get_available_splits(cls):
        return ["split0", "split1"]

    def __init__(self, samples, batchsize_per_replica=1):
        super().__init__(
            split=None,
            batchsize_per_replica=batchsize_per_replica,
            shuffle=False,
            transform=None,
            num_samples=len(samples),
        )
        input_tensors = [sample["input"] for sample in samples]
        target_tensors = [sample["target"] for sample in samples]
        self.dataset = ListDataset(input_tensors, target_tensors, loader=lambda x: x)

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        return cls(*args, **kwargs)


class TestRegistryFunctions(unittest.TestCase):
    """
    Tests functions that use registry
    """

    def test_build_model(self):
        dataset = build_dataset(DUMMY_CONFIG, DUMMY_SAMPLES_1)
        self.assertTrue(isinstance(dataset, TestDataset))

    def test_get_available_splits(self):
        # Test default classy splits
        splits = get_available_splits("test_dataset")
        self.assertEqual(splits, ["train", "test"])

    def test_get_available_splits_non_default(self):
        # Test default classy splits
        splits = get_available_splits("other_test_dataset")
        self.assertEqual(splits, ["split0", "split1"])


class TestClassyDataset(unittest.TestCase):
    """
    Tests member functions of ClassyDataset. Note, NotImplemented
    functions are mocked in TestDataset class.
    """

    def setUp(self):
        self.dataset1 = build_dataset(DUMMY_CONFIG, DUMMY_SAMPLES_1)
        self.dataset2 = build_dataset(DUMMY_CONFIG, DUMMY_SAMPLES_2)

    def _compare_samples(self, sample1, sample2):
        compare_samples(self, sample1, sample2)

    def _compare_batches(self, batch1, batch2):
        compare_batches(self, batch1, batch2)

    def test_init(self):
        self.assertTrue(self.dataset1 is not None)
        self.assertTrue(self.dataset2 is not None)

    def test_len(self):
        self.assertEqual(len(self.dataset1), 1)
        self.assertEqual(len(self.dataset2), 2)

    def test_getitem(self):
        sample = self.dataset1[0]
        self._compare_samples(sample, DUMMY_SAMPLES_1[0])

        for idx in range(len(self.dataset2)):
            sample = self.dataset2[idx]
            self._compare_samples(sample, DUMMY_SAMPLES_2[idx])

    def test_get_set_classy_state(self):
        state = self.dataset1.get_classy_state()
        self.assertEqual(
            state["wrapped_state"], self.dataset1.dataset.get_classy_state()
        )

        new_config = DUMMY_CONFIG.copy()
        new_config["dummy2"] = 2
        state["config"] = new_config
        self.dataset1.set_classy_state(state)

        # Check assert for changing dataset types
        with self.assertRaises(AssertionError):
            other_dataset = build_dataset(OTHER_DUMMY_CONFIG, DUMMY_SAMPLES_1)
            other_dataset.set_classy_state(state)

        # Verify when strict flag is false, this does not throw
        other_dataset = build_dataset(OTHER_DUMMY_CONFIG, DUMMY_SAMPLES_1)
        other_dataset.set_classy_state(state, strict=False)

    def test_get_iterator(self):
        # Verifies that we can retrieve samples with iterators
        dl = self.dataset1.iterator(num_workers=0)
        assert isinstance(
            dl, DataLoader
        ), "Classy Iterator should return instance of PyTorch Dataloader"
        next(iter(dl))

        dl = self.dataset1.iterator(num_workers=2)
        assert isinstance(
            dl, DataLoader
        ), "Classy Iterator should return instance of PyTorch Dataloader"
        it = iter(dl)
        next(it)
        # Because we use multiprocessing we delete the iterable to
        # shutdown workers
        del it

    def test_batch_logic(self):
        dataset = TestDataset(DUMMY_SAMPLES_2, batchsize_per_replica=2)
        dl = dataset.iterator(num_workers=0)
        batch = next(iter(dl))
        self.assertEqual(batch["input"].size()[0], 2)
        self._compare_batches(batch, BATCHED_DUMMY_SAMPLES_2[0])

    @mock.patch(
        "classy_vision.dataset.classy_dataset.get_world_size", mock_get_world_size
    )
    @mock.patch("classy_vision.dataset.classy_dataset.get_rank", mock_get_rank)
    def test_shard_logic(self):
        # This test uses a world size of 2, rank 1 to verify that the
        # second sample is returned by the dataloader
        dataset = TestDataset(DUMMY_SAMPLES_2, batchsize_per_replica=1)
        dl = dataset.iterator(num_workers=0)
        sample = next(iter(dl))
        self._compare_batches(sample, DUMMY_SAMPLES_2[1])

    def test_num_samples_logic(self):
        dataset = TestDataset(DUMMY_SAMPLES_2, num_samples=1)
        self.assertEqual(len(dataset), 1)

        # Check assert for num_samples > length of base dataset
        dataset = TestDataset(DUMMY_SAMPLES_2, num_samples=3)
        with self.assertRaises(AssertionError):
            len(dataset)

    def test_transform_logic(self):
        def _return_1_transform(sample):
            return 1

        dataset = TestDataset(DUMMY_SAMPLES_2, transform=_return_1_transform)
        sample = dataset[0]
        self.assertEqual(sample, 1)
