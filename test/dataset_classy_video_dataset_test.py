#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from classy_vision.dataset import build_dataset, register_dataset
from classy_vision.dataset.classy_video_dataset import ClassyVideoDataset
from classy_vision.dataset.core import ListDataset
from classy_vision.dataset.transforms.util_video import (
    build_video_field_transform_default,
)


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
        frames_per_clip,
        video_width,
        video_height,
        video_min_dimension,
        audio_samples,
        step_between_clips,
        frame_rate,
        clips_per_video,
        samples,
    ):
        super(TestVideoDataset, self).__init__(
            split,
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            frames_per_clip,
            video_width,
            video_height,
            video_min_dimension,
            audio_samples,
            step_between_clips,
            frame_rate,
            clips_per_video,
        )
        self.samples = samples
        input_tensors = [sample["input"] for sample in samples]
        target_tensors = [sample["target"] for sample in samples]
        self.dataset = ListDataset(input_tensors, target_tensors, loader=lambda x: x)

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
            frames_per_clip,
            video_width,
            video_height,
            video_min_dimension,
            audio_samples,
            step_between_clips,
            frame_rate,
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
            frames_per_clip,
            video_width,
            video_height,
            video_min_dimension,
            audio_samples,
            step_between_clips,
            frame_rate,
            clips_per_video,
        ) = self.dataset.parse_config(DUMMY_CONFIG)
