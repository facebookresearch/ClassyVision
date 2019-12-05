#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest

import PIL
from classy_vision.dataset import build_dataset
from classy_vision.dataset.classy_imagenet import ImageNetDataset
from torchvision import transforms


class TestImageNet(unittest.TestCase):
    def get_test_image_dataset(self, num_samples):
        config = {
            "name": "synthetic_image",
            "crop_size": 224,
            "num_channels": 3,
            "seed": 0,
            "class_ratio": 0.5,
            "num_samples": num_samples,
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
        for split in ["train", "val"]:
            os.makedirs(f"{self.base_dir}/{split}/0")
            os.makedirs(f"{self.base_dir}/{split}/1")

    def tearDown(self):
        # delete all the temporary data created
        shutil.rmtree(self.base_dir)

    def test_imagenet_retrieve_sample(self):
        num_samples = 10
        for split in ["train", "val"]:
            dataloader = self.get_test_image_dataset(num_samples).iterator()
            for i, sample in enumerate(dataloader):
                input = sample["input"]
                target = sample["target"]
                image = transforms.ToPILImage()(input.squeeze())
                path = f"{self.base_dir}/{split}/{target.item()}/{i}.png"
                # save the image in a lossless format (png)
                image.save(path)

            dataset = ImageNetDataset(
                split="train",
                batchsize_per_replica=1,
                shuffle=True,
                transform=None,
                num_samples=None,
                root=self.base_dir,
            )
            self.assertEqual(len(dataset), num_samples)
            img, target = dataset[0]
            self.assertTrue(isinstance(img, PIL.Image.Image))
            self.assertTrue(isinstance(target, int))
