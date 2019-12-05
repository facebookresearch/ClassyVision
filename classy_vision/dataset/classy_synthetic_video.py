#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict

from classy_vision.generic.distributed_util import get_rank, get_world_size
from torch.utils.data.distributed import DistributedSampler

from . import register_dataset
from .classy_video_dataset import ClassyVideoDataset
from .core import RandomVideoDataset
from .transforms.util_video import build_video_field_transform_default


@register_dataset("synthetic_video")
class SyntheticVideoDataset(ClassyVideoDataset):
    """Classy Dataset which produces random synthetic video clips.

    Useful for testing since the dataset is much faster to initialize and fetch samples
    from, compared to real world datasets.

    Note: Unlike :class:`SyntheticImageDataset`, this dataset generates targets
        randomly, independent of the video clips.
    """

    def __init__(
        self,
        num_classes: int,
        split: str,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Callable,
        num_samples: int,
        frames_per_clip: int,
        video_width: int,
        video_height: int,
        audio_samples: int,
        clips_per_video: int,
    ):
        """The constructor of SyntheticVideoDataset.

        Args:
            num_classes: Number of classes in the generated targets.
            split: Split of dataset to use
            batchsize_per_replica: batch size per model replica
            shuffle: Whether we should shuffle between epochs
            transform: Transform to be applied to each sample
            num_samples: Number of samples to return
            frames_per_clip: Number of frames in a video clip
            video_width: Width of the video clip
            video_height: Height of the video clip
            audio_samples: Audio sample rate
            clips_per_video: Number of clips per video
        """
        dataset = RandomVideoDataset(
            num_classes,
            split,
            num_samples,
            frames_per_clip,
            video_width,
            video_height,
            audio_samples,
            clips_per_video,
        )
        super().__init__(
            dataset,
            split,
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            clips_per_video,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SyntheticVideoDataset":
        """Instantiates a SyntheticVideoDataset from a configuration.

        Args:
            config: A configuration for a SyntheticVideoDataset.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SyntheticVideoDataset instance.
        """
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
            audio_samples,
            clips_per_video,
        )

    @property
    def video_clips(self):
        raise NotImplementedError()

    def _get_sampler(self, epoch):
        world_size = get_world_size()
        rank = get_rank()
        sampler = DistributedSampler(
            self, num_replicas=world_size, rank=rank, shuffle=self.shuffle
        )
        sampler.set_epoch(epoch)
        return sampler
