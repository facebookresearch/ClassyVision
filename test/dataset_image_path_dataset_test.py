#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest

import torch
from classy_vision.dataset import ClassyDataset, build_dataset
from classy_vision.dataset.image_path_dataset import ImagePathDataset
from torchvision import transforms


def _sort_key(x):
    # sorts x which could correspond to either (sample["input"], sample["target"]),
    # or sample["input"]
    if isinstance(x, tuple):
        return x[0].tolist() + x[1].tolist()
    else:
        return x.tolist()


class TestImageDataset(unittest.TestCase):
    def get_test_image_dataset(self):
        config = {
            "name": "synthetic_image",
            "crop_size": 224,
            "num_channels": 3,
            "seed": 0,
            "class_ratio": 0.5,
            "num_samples": 100,
            "batchsize_per_replica": 1,
            "use_shuffle": False,
            "transforms": [
                {
                    "name": "apply_transform_to_key",
                    "transforms": [{"name": "ToTensor"}],
                    "key": "input",
                }
            ],
        }
        dataset = build_dataset(config)
        return dataset

    def setUp(self):
        # create a base directory to write image files to
        self.base_dir = tempfile.mkdtemp()

        # create a dir to store images in the torchvision.ImageFolder format
        self.torchvision_dir = f"{self.base_dir}/tv"
        os.mkdir(self.torchvision_dir)
        os.mkdir(f"{self.torchvision_dir}/0")
        os.mkdir(f"{self.torchvision_dir}/1")

        # create a dir to store images in the other format
        self.other_dir = f"{self.base_dir}/other"
        os.mkdir(self.other_dir)

    def tearDown(self):
        # delete all the temporary data created
        shutil.rmtree(self.base_dir)

    def get_dataset_config(self):
        return {
            "name": "image_path",
            "batchsize_per_replica": 1,
            "use_shuffle": False,
            "num_samples": None,
            "transforms": [
                {
                    "name": "apply_transform_to_key",
                    "transforms": [{"name": "ToTensor"}],
                    "key": "input",
                }
            ],
        }

    def test_build_dataset(self):
        config = self.get_dataset_config()
        config["image_files"] = ["abc"]
        dataset = build_dataset(config)
        self.assertIsInstance(dataset, ClassyDataset)

        # test invalid configs

        # cannot pass both image_files and image_folder
        config["image_folder"] = self.torchvision_dir
        with self.assertRaises(Exception):
            dataset = build_dataset(config)

        # cannot skip both image_files and image_folder
        config.pop("image_files")
        config.pop("image_folder")
        with self.assertRaises(Exception):
            dataset = build_dataset(config)

    def test_image_dataset(self):
        image_files = []
        inputs = []
        targets = {}
        dataloader = self.get_test_image_dataset().iterator()
        for i, sample in enumerate(dataloader):
            input = sample["input"]
            target = sample["target"]
            image = transforms.ToPILImage()(input.squeeze())
            path = f"{self.torchvision_dir}/{target.item()}/{i}.png"
            image_files.append(path)
            image.save(path)
            path = f"{self.other_dir}/{i}.png"
            image.save(path)
            inputs.append(input)
            targets[input] = target

        config = self.get_dataset_config()

        config["image_files"] = image_files
        # test the dataset using image_files
        dataset = ImagePathDataset.from_config(config)
        dataloader = dataset.iterator()
        # the samples should be in the same order
        for sample, expected_input in zip(dataloader, inputs):
            self.assertTrue(torch.allclose(sample["input"], expected_input))
        config.pop("image_files")

        # test the dataset with image_folder of type torchvision.ImageFolder
        config["image_folder"] = self.torchvision_dir
        dataset = ImagePathDataset.from_config(config)
        dataloader = dataset.iterator()
        # the order doesn't matter, so we sort the results
        # note that this test assumes that the target for directory 0 will be 0
        for (input, target), (expected_input, expected_target) in zip(
            sorted(
                ((sample["input"], sample["target"]) for sample in dataloader),
                key=_sort_key,
            ),
            sorted(targets.items(), key=_sort_key),
        ):
            self.assertTrue(torch.allclose(input, expected_input))
            self.assertEqual(target, expected_target)

        # test the dataset with image_folder of the second type
        config["image_folder"] = self.other_dir
        dataset = ImagePathDataset.from_config(config)
        dataloader = dataset.iterator()
        # the order doesn't matter, so we sort the results
        for input, expected_input in zip(
            sorted((sample["input"] for sample in dataloader), key=_sort_key),
            sorted(inputs, key=_sort_key),
        ):
            self.assertTrue(torch.allclose(input, expected_input))
