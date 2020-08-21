#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
import torchvision.transforms as transforms
from classy_vision.dataset.dataloader_async_gpu_wrapper import DataloaderAsyncGPUWrapper
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ZeroImageDataset(Dataset):
    def __init__(self, crop_size, num_channels, num_classes, num_samples):
        self.crop_size = crop_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_samples = num_samples

    def __iter__(self):
        # Spread work as mod(N)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return iter(range(self.num_samples))
        else:
            return iter(
                range(worker_info.id, self.num_samples, worker_info.num_workers)
            )

    def __getitem__(self, index):
        input_data = transforms.ToTensor()(
            Image.fromarray(
                np.zeros(
                    (self.crop_size, self.crop_size, self.num_channels), dtype=np.uint8
                )
            )
        )
        target = np.random.randint(self.num_classes)
        return {"input": input_data, "target": target, "id": index}

    def __len__(self):
        return self.num_samples


class TestDataloaderAsyncGPUWrapper(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "This test needs a gpu to run")
    def test_dataset_async(self):
        """
        Test that streaming datasets return the correct number of batches, and that
        the length is also calculated correctly.
        """

        NUM_SAMPLES = 1024
        dataset = ZeroImageDataset(
            crop_size=224, num_channels=3, num_classes=1000, num_samples=NUM_SAMPLES
        )

        base_dataloader = DataLoader(dataset=dataset, pin_memory=True, num_workers=20)
        dataloader = DataloaderAsyncGPUWrapper(base_dataloader)

        # Test wrap correctness
        i = 0
        for sample in dataloader:
            # test that the data being served is all zeros
            self.assertTrue(sample["input"].nonzero(as_tuple=False).numel() == 0)

            # test that it's all cuda tensors
            for k in sample.keys():
                self.assertTrue(sample[k].device.type == "cuda")

            # check that consecutive samples are independent objects in memory
            sample["input"].fill_(3.14)

            # check that the expected number of samples is served
            i += 1
        self.assertEqual(i, NUM_SAMPLES)
