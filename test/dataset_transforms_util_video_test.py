#!/usr/bin/env python3

import copy
import unittest

import torch
import torchvision.transforms as transforms
from classy_vision.dataset.core.random_video_datasets import RandomVideoDataset
from classy_vision.dataset.transforms.util_video import (
    VideoConstants,
    build_video_field_transform_default,
)


class DatasetTransformUtilVideoTest(unittest.TestCase):
    def get_test_video_dataset(self):
        self.config = {
            "num_channels": 3,
            "num_frames": 32,
            "height": 128,
            "width": 160,
            "sample_rate": 44100,
            "num_classes": 10,
            "seed": 0,
            "num_samples": 100,
        }
        dataset = RandomVideoDataset(self.config)
        return dataset

    def test_build_field_transform_default_video(self):
        dataset = self.get_test_video_dataset()

        # transform config is not provided. Use default transforms
        config = None
        # default training data transform
        sample = dataset[0]

        transform = build_video_field_transform_default(config, "train")
        output_clip = transform(sample)["input"]["video"]
        self.assertTrue(output_clip.size(0) == self.config["num_channels"])
        self.assertTrue(output_clip.size(1) == self.config["num_frames"])
        self.assertTrue(output_clip.size(2) == VideoConstants.CROP_SIZE)
        self.assertTrue(output_clip.size(3) == VideoConstants.CROP_SIZE)

        # default testing data transform
        sample = dataset[1]
        sample_copy = copy.deepcopy(sample)

        expected_output_clip = transforms.ToTensorVideo()(sample["input"]["video"])
        expected_output_clip = transforms.CenterCropVideo(VideoConstants.CROP_SIZE)(
            expected_output_clip
        )
        expected_output_clip = transforms.NormalizeVideo(
            mean=VideoConstants.MEAN, std=VideoConstants.STD
        )(expected_output_clip)

        transform = build_video_field_transform_default(config, "test")
        output_clip = transform(sample_copy)["input"]["video"]

        self.assertTrue(torch.all(torch.eq(expected_output_clip, output_clip)))

        # transform config is provided.
        sample = dataset[2]

        config = {
            "video": [
                {"name": "ToTensorVideo"},
                {"name": "RandomResizedCropVideo", "size": 64},
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
        self.assertTrue(output_clip.size(0) == self.config["num_channels"])
        self.assertTrue(output_clip.size(1) == self.config["num_frames"])
        self.assertTrue(output_clip.size(2) == 64)
        self.assertTrue(output_clip.size(3) == 64)
        self.assertTrue(output_clip.dtype == torch.float)
