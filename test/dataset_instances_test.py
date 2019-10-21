#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
from test.generic.utils import make_torch_deterministic

from classy_vision.dataset import build_dataset, get_available_splits
from classy_vision.dataset.classy_dataset import ClassyDataset


# list of datasets expected to be present:
DEFAULT_SETTINGS = {
    "batchsizes_per_replica": [1, 2],
    "num_samples": [10],
    "shuffle": [True, False],
}

RAPID_DATASETS_TO_TEST = [
    "imagenet",
    # TODO: T48865995
    # "cub2011",
    # "coco",
    # "omniglot",
    # "oxford_flowers",
    # "pascal_voc2007",
    # "pascal_voc2007_ml",
    # "pascal_voc2012",
    # "pascal_voc2012_ml",
    # "sun397",
]

SLOW_DATASETS_TO_TEST = [
    "cifar10",  # Slow test, 10m
    "cifar100",  # Slow test, 10m
    #  'uru',  # T44459960,
    "oc_nudity_q1_violates",  # VERY slow test, >60m
    "oc_nudity_q1_benign",  # VERY slow test, >60m
    "oc_nudity_q1_suggestive",  # VERY slow test, >60m
    "oc_violence_benign_vs_mad",  # VERY slow test, >60m
    "places365",  # Slow test, 6m
    "svhn",  # Slow test, 10m
    "yfcc100m",  # VERY Slow test, >60m
]

MAX_NUM_SAMPLES = 5


class TestDatasets(unittest.TestCase):
    """
    Tests all implemented datasets (ImageNet, CIFAR-x, etc.) to verify
    that samples are fetched as expected. These are integration tests
    that may rely on file systems or other non-classy-vision external
    systems. As such we don't run these by default on sandcastle, they
    have to be ran on demand by the user. We have to be careful with
    everstore datasets where we expect there to be missing samples

    All dataset wrappers should be tested in a unittest with dummy
    data, so we do not test that the wrappers all work here.

    This only tests the first MAX_NUM_SAMPLES and then a random set of
    MAX_NUM_SAMPLES when shuffle is enabled, but it does this multiple times
    """

    def setUp(self):
        make_torch_deterministic(0)
        self.configs = None

    def _generate_configs(self):
        """Memoized generation of configs for tests in a single list"""
        if self.configs is not None:
            return self.configs

        self.configs = []
        for batchsize_per_replica in DEFAULT_SETTINGS["batchsizes_per_replica"]:
            for num_samples in DEFAULT_SETTINGS["num_samples"]:
                for use_shuffle in DEFAULT_SETTINGS["shuffle"]:
                    self.configs.append(
                        {
                            "batchsize_per_replica": batchsize_per_replica,
                            "num_samples": num_samples,
                            "use_shuffle": use_shuffle,
                        }
                    )
        return self.configs

    def _check_dataset(self, dataset, dataset_name):
        # check dataset type and size:
        self.assertIsInstance(
            dataset,
            ClassyDataset,
            msg="incorrect dataset type for {dataset_name}: {dataset_type}, ".format(
                dataset_name=dataset_name, dataset_type=type(dataset)
            ),
        )
        self.assertGreater(
            len(dataset), 0, msg="incorrect dataset size: {}".format(dataset_name)
        )

        # check subset of samples in datasets (including first and last):
        perm = [idx for idx in range(0, min(MAX_NUM_SAMPLES, len(dataset)))]
        if perm[-1] != len(dataset) - 1:
            perm.append(len(dataset) - 1)
        for idx in perm:
            sample = dataset[idx]
            if sample is not None:
                self.assertIsInstance(
                    sample,
                    dict,
                    msg="incorrect sample type for {dataset_name}:"
                    "{sample_type}".format(
                        dataset_name=dataset_name, sample_type=type(sample)
                    ),
                )
                # We can have datasets without targets or sharding. Also,
                # Everstore datasets can error out retrieving empty
                # samples
            self.assertTrue(sample is None or "input" in sample or len(sample) == 0)

    @unittest.skipUnless(
        False, "These tests rely on gluster / other systems, run on devserver"
    )
    def test_dataset_rapid(self):
        # loop over all datasets:
        for dataset_name in RAPID_DATASETS_TO_TEST:
            for config in self._generate_configs():
                for split in get_available_splits(dataset_name):
                    config["name"] = dataset_name
                    config["split"] = split
                    dataset = build_dataset(config)
                    if (
                        hasattr(self.dataloaders[self.phase_type].dataset, "do_shuffle")
                        and config["use_shuffle"]
                    ):
                        dataset.do_shuffle(epoch_num=0)

                    self._check_dataset(dataset, dataset_name)

    @unittest.skipUnless(
        False,  # These tests are slow, run only if True
        "These tests rely on gluster / other systems and are very slow",
    )
    def test_dataset_slow(self):
        # loop over all datasets:
        for dataset_name in SLOW_DATASETS_TO_TEST:
            for config in self._generate_configs():
                for split in get_available_splits(dataset_name):
                    config["name"] = dataset_name
                    config["split"] = split
                    dataset = build_dataset(config)
                    if (
                        hasattr(self.dataloaders[self.phase_type].dataset, "do_shuffle")
                        and config["use_shuffle"]
                    ):
                        dataset.do_shuffle(epoch_num=0)

                    self._check_dataset(dataset, dataset_name)


# run all the tests:
if __name__ == "__main__":
    unittest.main(argv=sys.argv[0])
