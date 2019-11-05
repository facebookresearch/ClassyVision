#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os

import torch
from classy_vision.generic.distributed_util import get_rank, get_world_size
from torchvision.datasets.samplers.clip_sampler import (
    DistributedSampler,
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
        dataset,
        split,
        batchsize_per_replica,
        shuffle,
        transform,
        num_samples,
        clips_per_video,
    ):
        super(ClassyVideoDataset, self).__init__(
            dataset, split, batchsize_per_replica, shuffle, transform, num_samples
        )
        # Assignments:
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
            # At testing time, we do not crop frames but conduct a FCN-style evaluation.
            # Video spatial resolution can vary from video to video. So we test one
            # video at a time, and NO. of clips in a minibatch should be equal to
            # No. of clips sampled from a video
            if not batchsize_per_replica == clips_per_video:
                logging.warning(
                    f"For testing, batchsize per replica ({batchsize_per_replica})"
                    + f"should be equal to clips_per_video ({clips_per_video})"
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

    @classmethod
    def load_metadata(cls, filepath, video_dir=None, update_file_path=False):
        metadata = torch.load(filepath)
        if video_dir is not None and update_file_path:
            # video path in meta data can be computed in a different root video folder
            # If we use a different root video folder, we need to update the video paths
            assert os.path.exists(video_dir), "folder does not exist: %s" % video_dir
            for idx, video_path in enumerate(metadata["video_paths"]):
                # video path template is $VIDEO_DIR/$CLASS_NAME/$VIDEO_FILE
                dirname, filename = os.path.split(video_path)
                _, class_name = os.path.split(dirname)
                metadata["video_paths"][idx] = os.path.join(
                    video_dir, class_name, filename
                )
        return metadata

    @classmethod
    def save_metadata(cls, metadata, filepath):
        filedir = os.path.dirname(filepath)
        if not os.path.exists(filedir):
            try:
                os.mkdirs(filedir)
            except Exception as err:
                logging.warn(f"Fail to create folder: {filedir}")
                raise err
        logging.info(f"Save metadata to file: {filedir}")
        try:
            torch.save(metadata, filepath)
        except ValueError:
            logging.warn(f"Fail to save metadata to file: {filepath}")

    def _get_sampler(self, epoch):
        if self.split == "train":
            # For video model training, we don't necessarily want to use all possible
            # clips in the video in one training epoch. More often, we randomly
            # sample at most N clips per training video. In practice, N is often 1
            clip_sampler = RandomClipSampler(
                self.dataset.video_clips, self.clips_per_video
            )
        else:
            # For video model testing, we sample N evenly spaced clips per test
            # video. We will simply average predictions over them
            clip_sampler = UniformClipSampler(
                self.dataset.video_clips, self.clips_per_video
            )
        world_size = get_world_size()
        rank = get_rank()
        sampler = DistributedSampler(
            clip_sampler,
            num_replicas=world_size,
            rank=rank,
            shuffle=self.shuffle,
            group_size=self.clips_per_video,
            num_samples=self.num_samples,
        )
        sampler.set_epoch(epoch)
        return sampler

    def iterator(self, *args, **kwargs):
        # for video dataset, it may use VideoClips class from TorchVision,
        # which further use a cpp python extension for video decoding.
        # It is difficult to use "spawning" as multiprocessing start method.
        # Thus we choose "fork" as multiprocessing start method.
        if "num_workers" in kwargs and kwargs["num_workers"] > 0:
            mp = multiprocessing.get_context("fork")
            kwargs["multiprocessing_context"] = mp
        return super(ClassyVideoDataset, self).iterator(*args, **kwargs)
