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
        os.mkdir(f"{self.base_dir}/0")
        os.mkdir(f"{self.base_dir}/1")

    def tearDown(self):
        # delete all the temporary data created
        shutil.rmtree(self.base_dir)

    def get_dataset_config(self):
        return {
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

    @unittest.skip(
        "Skipping test since build_dataset doesn't "
        "work right now for ImagePathDataset"
    )
    def test_build_dataset(self):
        config = self.get_dataset_config()
        dataset = build_dataset(config)
        self.assertIsInstance(dataset, ClassyDataset)

    def test_image_dataset(self):
        image_paths = []
        inputs = []
        targets = []
        dataloader = self.get_test_image_dataset().iterator()
        for i, sample in enumerate(dataloader):
            input = sample["input"]
            target = sample["target"]
            image = transforms.ToPILImage()(input.squeeze())
            path = f"{self.base_dir}/{target.item()}/{i}.png"
            # save the image in a lossless format (png)
            image.save(path)
            image_paths.append(path)
            inputs.append(input)
            targets.append(target)

        # config for the image dataset
        config = self.get_dataset_config()

        # create an image dataset from the list of images
        dataset = ImagePathDataset.from_config(
            config, image_paths=image_paths, targets=targets
        )
        dataloader = dataset.iterator()
        # the samples should be in the same order
        for sample, expected_input, expected_target in zip(dataloader, inputs, targets):
            self.assertTrue(torch.allclose(sample["input"], expected_input))
            self.assertEqual(sample["target"], expected_target)

        # test the dataset works without targets as well
        dataset = ImagePathDataset.from_config(config, image_paths=image_paths)
        dataloader = dataset.iterator()
        # the samples should be in the same order
        for sample, expected_input in zip(dataloader, inputs):
            self.assertTrue(torch.allclose(sample["input"], expected_input))

        # create an image dataset from the root dir
        dataset = ImagePathDataset.from_config(config, image_paths=self.base_dir)
        dataloader = dataset.iterator()
        # test that we get the same class distribution
        # we don't test the actual samples since the ordering isn't defined
        counts = [0, 0]
        for sample in dataloader:
            counts[sample["target"].item()] += 1
        expected_counts = [0, 0]
        for target in targets:
            expected_counts[target.item()] += 1
        self.assertEqual(counts, expected_counts)
