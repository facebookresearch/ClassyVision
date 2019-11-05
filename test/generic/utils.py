#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import wraps

import torch


class Arguments(object):
    """Object that looks like input arguments. Used to spoof argparse namespace."""

    def __init__(self, **args):
        self.args = args
        self.__dict__.update(args)

    def __iter__(self):
        return iter(self.args)

    def __eq__(self, other):
        if isinstance(other, Arguments):
            return self.args == other.args
        else:
            return NotImplemented

    def _asdict(self):
        return vars(self)


def skip_if_no_gpu(func):
    """Decorator that can be used to skip GPU tests on non-GPU machines."""
    func.skip_if_no_gpu = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return
        if torch.cuda.device_count() <= 0:
            return

        return func(*args, **kwargs)

    return wrapper


def repeat_test(original_function=None, *, num_times=3):
    """Decorator that can be used to repeat test multiple times."""

    def repeat_test_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                func(*args, **kwargs)

        return wrapper

    # this handles default arguments to decorator:
    if original_function:
        return repeat_test_decorator(original_function)
    return repeat_test_decorator


def make_torch_deterministic(seed=0):
    """Makes Torch code run deterministically."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def compare_batches(test_fixture, batch1, batch2):
    """Compare two batches. Does not do recursive comparison"""
    test_fixture.assertEqual(type(batch1), type(batch2))
    if isinstance(batch1, (tuple, list)):
        test_fixture.assertEqual(len(batch1), len(batch2))
        for n in range(len(batch1)):
            value1 = batch1[n]
            value2 = batch2[n]
            test_fixture.assertEqual(type(value1), type(value2))
            if torch.is_tensor(value1):
                test_fixture.assertTrue(torch.allclose(value1, value2))
            else:
                test_fixture.assertEqual(value1, value2)

    elif isinstance(batch1, dict):
        test_fixture.assertEqual(batch1.keys(), batch2.keys())
        for key, value1 in batch1.items():
            value2 = batch2[key]
            test_fixture.assertEqual(type(value1), type(value2))
            if torch.is_tensor(value1):
                test_fixture.assertTrue(torch.allclose(value1, value2))
            else:
                test_fixture.assertEqual(value1, value2)


def compare_datasets(test_fixture, dataset1, dataset2):
    test_fixture.assertEqual(len(dataset1), len(dataset2))
    for idx in range(len(dataset1)):
        compare_batches(test_fixture, dataset1[idx], dataset2[idx])


def compare_batchlist_and_dataset_with_skips(
    test_fixture, batch_list, dataset, skip_indices=None
):
    """
    Compares a list of batches and the dataset.  If some samples were
    skipped in the iterator (i.e. if we simulated an error on that
    sample), that should be indicated in the skip_indices list
    """
    if skip_indices is None:
        skip_indices = []
    if isinstance(skip_indices, int):
        skip_indices = [skip_indices]

    skips = 0
    for idx, batch in enumerate(batch_list):
        while (idx + skips) in skip_indices:
            skips += 1
        dataset_batch = dataset[idx + skips]
        compare_batches(test_fixture, batch, dataset_batch)


class MockErrorDataset:
    """
    Dataset used for testing. Wraps a real dataset with a
    batchsize_per_replica, but allows us to delete samples on return
    to simulate errors (similar to what happens with Everstore)
    """

    def __init__(self, dataset):
        self.rebatch_map = {}
        self.dataset = dataset
        self.batchsize_per_replica = dataset.batchsize_per_replica

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        # If rebatch map contains index, resize the batch
        if idx in self.rebatch_map:
            num_samples = self.rebatch_map[idx]
            if num_samples < batch["input"].size()[0]:
                batch["input"] = batch["input"][:num_samples]
                batch["target"] = batch["target"][:num_samples]

        return batch

    def __len__(self):
        return len(self.dataset)


def recursive_unpack(batch):
    """
    Takes a batch of samples, e.g.

      batch = {'input': tensor([256, 3, 224, 224]), 'target': tensor([256])}

    and unpacks them into a list of single samples, e.g.

      [{'input': tensor([1, 3, 224, 224]), 'target': tensor([1])} ... ]
    """
    new_list = []
    if isinstance(batch, dict):
        unpacked_dict = {}
        batchsize_per_replica = -1
        for key, val in batch.items():
            unpacked_dict[key] = recursive_unpack(val)
            batchsize_per_replica = (
                len(unpacked_dict[key])
                if not torch.is_tensor(unpacked_dict[key])
                else 1
            )

        for idx in range(batchsize_per_replica):
            sample = {}
            for key, val in unpacked_dict.items():
                sample[key] = val[idx]

            new_list.append(sample)
        return new_list

    elif isinstance(batch, (list, tuple)):
        unpacked_list = []
        if isinstance(batch, tuple):
            batch = list(batch)

        for val in batch:
            unpacked_list.append(recursive_unpack(val))
            batchsize_per_replica = (
                len(unpacked_list[0]) if not torch.is_tensor(unpacked_list[0]) else 1
            )

        for idx in range(batchsize_per_replica):
            sample = []
            for val in unpacked_list:
                sample.append(val[idx])

            if isinstance(batch, tuple):
                sample = tuple(sample)
            new_list.append(sample)
        return new_list

    elif torch.is_tensor(batch):
        for i in range(batch.size()[0]):
            new_list.append(batch[i])
        return new_list

    raise TypeError("Unexpected type %s passed to unpack" % type(batch))


def compare_model_state(test_fixture, state, state2, check_heads=True):
    for k in state["model"]["trunk"].keys():
        if not torch.allclose(state["model"]["trunk"][k], state2["model"]["trunk"][k]):
            print(k, state["model"]["trunk"][k], state2["model"]["trunk"][k])
        test_fixture.assertTrue(
            torch.allclose(state["model"]["trunk"][k], state2["model"]["trunk"][k])
        )
    if check_heads:
        for block, head_states in state["model"]["heads"].items():
            for head_id, states in head_states.items():
                for k in states.keys():
                    test_fixture.assertTrue(
                        torch.allclose(
                            state["model"]["heads"][block][head_id][k],
                            state2["model"]["heads"][block][head_id][k],
                        )
                    )


def compare_samples(test_fixture, sample1, sample2):
    test_fixture.assertEqual(sample1.keys(), sample2.keys())
    test_fixture.assertTrue(torch.is_tensor(sample1["input"]))
    test_fixture.assertTrue(torch.is_tensor(sample2["input"]))
    test_fixture.assertTrue(torch.is_tensor(sample1["target"]))
    test_fixture.assertTrue(torch.is_tensor(sample2["target"]))

    test_fixture.assertTrue(torch.allclose(sample1["input"], sample2["input"]))
    test_fixture.assertTrue(torch.allclose(sample1["target"], sample2["target"]))


def compare_states(test_fixture, state_1, state_2, check_heads=True):
    """
    Tests the classy state dicts for equality, but skips the member objects
    which implement their own {get, set}_classy_state functions.
    """
    # check base_model
    compare_model_state(
        test_fixture, state_1["base_model"], state_2["base_model"], check_heads
    )
    # check losses
    test_fixture.assertEqual(len(state_1["losses"]), len(state_2["losses"]))
    for loss_1, loss_2 in zip(state_1["losses"], state_2["losses"]):
        test_fixture.assertAlmostEqual(loss_1, loss_2)

    for key in ["base_model", "meters", "optimizer", "losses"]:
        # we trust that these have been tested using their unit tests or
        # by the code above
        test_fixture.assertIn(key, state_1)
        test_fixture.assertIn(key, state_2)
        del state_1[key]
        del state_2[key]
    test_fixture.assertDictEqual(state_1, state_2)
