#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing

from torchvision.datasets.samplers.clip_sampler import (
    RandomClipSampler,
    UniformClipSampler,
)

from .classy_dataset import ClassyDataset


class ClassyVideoDataset(ClassyDataset):
    """
    Interface specifying what a Classy Vision video dataset can be expected to provide.
    """

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
    ):
        super(ClassyVideoDataset, self).__init__(
            split, batchsize_per_replica, shuffle, transform, num_samples
        )
        # Assignments:
        self.frames_per_clip = frames_per_clip
        self.video_width = video_width
        self.video_height = video_height
        self.video_min_dimension = video_min_dimension
        self.audio_samples = audio_samples
        self.step_between_clips = step_between_clips
        self.frame_rate = frame_rate
        self.clips_per_video = clips_per_video

    @classmethod
    def parse_config(cls, config):
        assert "frames_per_clip" in config, "frames_per_clip must be set"

        video_width = config.get("video_width", 0)
        video_height = config.get("video_height", 0)
        video_min_dimension = config.get("video_min_dimension", 0)
        audio_samples = config.get("audio_samples", 0)
        step_between_clips = config.get("step_between_clips", 1)
        frame_rate = config.get("frame_rate", None)
        clips_per_video = config.get("clips_per_video", 1)

        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = super().parse_config(config)

        if not config["split"] == "train":
            assert batchsize_per_replica % clips_per_video == 0, (
                "For video test dataset, the batchsize per replica must be a "
                "multiplier of No. of clips samepled from each video"
            )

        return (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
            config["frames_per_clip"],
            video_width,
            video_height,
            video_min_dimension,
            audio_samples,
            step_between_clips,
            frame_rate,
            clips_per_video,
        )

    def wrap_video_dataset(
        self,
        dataset,
        split,
        clips_per_video,
        transform,
        batchsize_per_replica,
        shuffle=True,
        subsample=None,
    ):
        if split == "train":
            # For video model training, we don't necessarily want to use all possible
            # clips in the video in one training epoch. More often, we randomly
            # sample at most N clips per training video. In practice, N is often 1
            sampler = RandomClipSampler(dataset.video_clips, clips_per_video)
        else:
            # For video model testing, we sample N evenly spaced clips per test
            # video. We will simply average predictions over them
            sampler = UniformClipSampler(dataset.video_clips, clips_per_video)
        dataset = dataset.resample(list(sampler))

        return self.wrap_dataset(
            dataset,
            transform=transform,
            batchsize_per_replica=batchsize_per_replica,
            shuffle=shuffle,
            subsample=subsample,
            shard_group_size=1 if split == "train" else clips_per_video,
        )

    def iterator(self, *args, **kwargs):
        # for video dataset, it may use VideoClips class from TorchVision,
        # which further use a cpp python extension for video decoding.
        # It is difficult to use "spawning" as multiprocessing start method.
        # Thus we choose "fork" as multiprocessing start method.
        if "num_workers" in kwargs and kwargs["num_workers"] > 0:
            mp = multiprocessing.get_context("fork")
            kwargs["multiprocessing_context"] = mp
        return super(ClassyVideoDataset, self).iterator(*args, **kwargs)
