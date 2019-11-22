#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.dataset import build_dataset, register_dataset
from classy_vision.dataset.classy_video_dataset import (
    ClassyVideoDataset,
    MaxLengthClipSampler,
)
from classy_vision.dataset.core import ListDataset
from classy_vision.dataset.transforms.util_video import (
    build_video_field_transform_default,
)
from torch.utils.data import Sampler


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
    "split": "train",
    "batchsize_per_replica": 1,
    "use_shuffle": True,
    "num_samples": 1,
    "frames_per_clip": 8,
    "video_dir": "dummy_video_dir",
}


class MockClipSampler(Sampler):
    def __init__(self, full_size=1000):
        self.full_size = full_size

    def __iter__(self):
        indices = list(range(self.full_size))
        return iter(indices)

    def __len__(self):
        return self.full_size


@register_dataset("test_video_dataset")
class TestVideoDataset(ClassyVideoDataset):
    """Test dataset for validating registry functions"""

    def __init__(
        self,
        split,
        batchsize_per_replica,
        shuffle,
        transform,
        num_samples,
        clips_per_video,
        samples,
    ):
        self.samples = samples
        input_tensors = [sample["input"] for sample in samples]
        target_tensors = [sample["target"] for sample in samples]
        dataset = ListDataset(input_tensors, target_tensors, loader=lambda x: x)
        super(TestVideoDataset, self).__init__(
            dataset,
            split,
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            clips_per_video,
        )

    @classmethod
    def from_config(cls, config, samples):
        split = config.get("split")
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
            frames_per_clip,
            video_width,
            video_height,
            video_min_dimension,
            audio_samples,
            step_between_clips,
            frame_rate,
            clips_per_video,
        ) = cls.parse_config(config)
        transform = build_video_field_transform_default(transform_config, split)
        return cls(
            split,
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            clips_per_video,
            samples,
        )


class TestRegistryFunctions(unittest.TestCase):
    """
    Tests functions that use registry
    """

    def test_build_dataset(self):
        dataset = build_dataset(DUMMY_CONFIG, DUMMY_SAMPLES_1)
        self.assertTrue(isinstance(dataset, TestVideoDataset))


class TestClassyVideoDataset(unittest.TestCase):
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
            frames_per_clip,
            video_width,
            video_height,
            video_min_dimension,
            audio_samples,
            step_between_clips,
            frame_rate,
            clips_per_video,
        ) = self.dataset.parse_config(DUMMY_CONFIG)

    def test_max_length_clip_sampler(self):
        clip_sampler = MockClipSampler(full_size=1000)
        clip_sampler = MaxLengthClipSampler(clip_sampler, num_samples=64)
        count = 0
        for _clip_index in iter(clip_sampler):
            count += 1
        self.assertEqual(count, 64)
        self.assertEqual(len(clip_sampler), 64)
