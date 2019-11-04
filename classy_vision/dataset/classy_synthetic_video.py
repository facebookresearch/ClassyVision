#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from classy_vision.generic.distributed_util import get_rank, get_world_size
from torch.utils.data.distributed import DistributedSampler

from . import register_dataset
from .classy_video_dataset import ClassyVideoDataset
from .core import RandomVideoDataset
from .transforms.util_video import build_video_field_transform_default


@register_dataset("synthetic_video")
class SyntheticVideoClassificationDataset(ClassyVideoDataset):
    """
    This synthetic video classification dataset class randomly generates video
    clip data and label on-the-fly. Compared with other realistc video datasets,
    such as `HMDB51Dataset` and `Kinetics400Dataset`, it is fast to initialize
    and is independent of actual video files. It is useful to speed up daily
    dev work other than dataset class.
    """

    @classmethod
    def get_available_splits(cls):
        return ["train", "val", "test"]

    def __init__(
        self,
        num_classes,
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
        super(SyntheticVideoClassificationDataset, self).__init__(
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

        self.dataset = RandomVideoDataset(
            num_classes,
            split,
            num_samples,
            frames_per_clip,
            video_width,
            video_height,
            audio_samples,
            clips_per_video,
        )

    def _get_sampler(self, epoch):
        world_size = get_world_size()
        rank = get_rank()
        sampler = DistributedSampler(
            self, num_replicas=world_size, rank=rank, shuffle=self.shuffle
        )
        sampler.set_epoch(epoch)
        return sampler

    @classmethod
    def from_config(cls, config):
        split = config["split"]
        num_classes = config["num_classes"]
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
            num_classes,
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
