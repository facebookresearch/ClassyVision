#!/usr/bin/env python3

import unittest

import torch
from classy_vision.dataset import build_dataset, register_dataset
from classy_vision.dataset.classy_video_dataset import ClassyVideoDataset
from classy_vision.dataset.core import ListDataset


DUMMY_SAMPLES_1 = [
    {
        "input": {
            "video": torch.randint(0, 256, (8, 3, 128, 128), dtype=torch.uint8),
            "audio": torch.rand(1000, 1, dtype=torch.float32),
        },
        "target": torch.tensor([[0]]),
    }
]


DUMMY_CONFIG = {
    "name": "test_video_dataset",
    "batchsize_per_replica": 1,
    "use_shuffle": True,
    "num_samples": 1,
    "frames_per_clip": 8,
    "video_dir": "dummy_video_dir",
}


@register_dataset("test_video_dataset")
class TestVideoDataset(ClassyVideoDataset):
    """Test dataset for validating registry functions"""

    def __init__(self, config, samples):
        super(TestVideoDataset, self).__init__(config)
        self.samples = samples
        input_tensors = [sample["input"] for sample in samples]
        target_tensors = [sample["target"] for sample in samples]
        self.dataset = ListDataset(input_tensors, target_tensors, loader=lambda x: x)


class TestRegistryFunctions(unittest.TestCase):
    """
    Tests functions that use registry
    """

    def test_build_dataset(self):
        dataset = build_dataset(DUMMY_CONFIG, DUMMY_SAMPLES_1)
        self.assertTrue(isinstance(dataset, TestVideoDataset))


class TestClassyDataset(unittest.TestCase):
    """
    Tests member functions of ClassyVideoDataset.
    """

    def setUp(self):
        self.dataset = build_dataset(DUMMY_CONFIG, DUMMY_SAMPLES_1)

    def test_parse_config(self):
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
            step_between_clips,
            frame_rate,
            precomputed_metadata_filepath,
            save_metadata_filepath,
        ) = self.dataset.parse_config(self.dataset._config)
