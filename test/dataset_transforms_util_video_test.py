#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
import torchvision.transforms._transforms_video as transforms_video
from classy_vision.dataset.core.random_video_datasets import RandomVideoDataset
from classy_vision.dataset.transforms.util_video import (
    VideoConstants,
    build_video_field_transform_default,
)


class DatasetTransformUtilVideoTest(unittest.TestCase):
    def get_test_video_dataset(self):
        self.num_classes = 10
        self.split = "train"
        self.num_samples = 100
        self.frames_per_clip = 32
        self.video_width = 320
        self.video_height = 256
        self.audio_samples = 44000
        self.clips_per_video = 1
        self.seed = 1

        dataset = RandomVideoDataset(
            self.num_classes,
            self.split,
            self.num_samples,
            self.frames_per_clip,
            self.video_width,
            self.video_height,
            self.audio_samples,
            self.clips_per_video,
            self.seed,
        )
        return dataset

    def test_build_field_transform_default_video(self):
        dataset = self.get_test_video_dataset()

        # transform config is not provided. Use default transforms
        config = None
        # default training data transform
        sample = dataset[0]

        transform = build_video_field_transform_default(config, "train")
        output_clip = transform(sample)["input"]["video"]
        self.assertEqual(
            output_clip.size(),
            torch.Size(
                (
                    3,
                    self.frames_per_clip,
                    VideoConstants.CROP_SIZE,
                    VideoConstants.CROP_SIZE,
                )
            ),
        )
        # default testing data transform
        sample = dataset[1]
        sample_copy = copy.deepcopy(sample)

        expected_output_clip = transforms_video.ToTensorVideo()(
            sample["input"]["video"]
        )
        expected_output_clip = transforms_video.CenterCropVideo(
            VideoConstants.CROP_SIZE
        )(expected_output_clip)
        expected_output_clip = transforms_video.NormalizeVideo(
            mean=VideoConstants.MEAN, std=VideoConstants.STD
        )(expected_output_clip)

        transform = build_video_field_transform_default(config, "test")
        output_clip = transform(sample_copy)["input"]["video"]

        rescaled_width = int(
            VideoConstants.SIZE_RANGE[0] * self.video_width / self.video_height
        )
        self.assertEqual(
            output_clip.size(),
            torch.Size(
                (3, self.frames_per_clip, VideoConstants.SIZE_RANGE[0], rescaled_width)
            ),
        )
        # transform config is provided. Simulate training config
        sample = dataset[2]
        config = {
            "video": [
                {"name": "ToTensorVideo"},
                {
                    "name": "video_clip_random_resize_crop",
                    "crop_size": 64,
                    "size_range": [256, 320],
                },
                {"name": "RandomHorizontalFlipVideo"},
                {
                    "name": "NormalizeVideo",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ]
        }
        transform = build_video_field_transform_default(config, "train")
        output_clip = transform(sample)["input"]["video"]
        self.assertEqual(
            output_clip.size(), torch.Size((3, self.frames_per_clip, 64, 64))
        )
        self.assertTrue(output_clip.dtype == torch.float)

        # transform config is provided. Simulate testing config
        sample = dataset[3]
        config = {
            "video": [
                {"name": "ToTensorVideo"},
                {"name": "video_clip_resize", "size": 64},
                {
                    "name": "NormalizeVideo",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ]
        }
        transform = build_video_field_transform_default(config, "train")
        output_clip = transform(sample)["input"]["video"]

        rescaled_width = int(64 * self.video_width / self.video_height)
        self.assertEqual(
            output_clip.size(),
            torch.Size((3, self.frames_per_clip, 64, rescaled_width)),
        )
        self.assertTrue(output_clip.dtype == torch.float)
