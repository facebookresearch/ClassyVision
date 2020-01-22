#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Dict, Optional

import torch
from torchvision.datasets.ucf101 import UCF101

from . import register_dataset
from .classy_video_dataset import ClassyVideoDataset
from .transforms.util_video import build_video_field_transform_default


@register_dataset("ucf101")
class UCF101Dataset(ClassyVideoDataset):
    """`UCF101 <https://www.crcv.ucf.edu/data/UCF101.php/>`_ is an action
    recognition video dataset, and it has 101 classes.

    It is built on top of `UCF101 <https://github.com/pytorch/vision/blob/master
    /torchvision/datasets/ucf101.py#L10>`_ dataset class in TorchVision.

    """

    def __init__(
        self,
        split: str,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Callable,
        num_samples: Optional[int],
        frames_per_clip: int,
        video_width: int,
        video_height: int,
        video_min_dimension: int,
        audio_samples: int,
        step_between_clips: int,
        frame_rate: Optional[int],
        clips_per_video: int,
        video_dir: str,
        splits_dir: str,
        fold: int,
        metadata_filepath: str,
    ):
        """The constructor of UCF101Dataset.

        Args:
            split: dataset split which can be either "train" or "test"
            batchsize_per_replica: batch size per model replica
            shuffle: If true, shuffle the dataset
            transform: a dict where transforms video and audio data
            num_samples: if not None, it will subsample dataset
            frames_per_clip: the No. of frames in a video clip
            video_width: rescaled video width. If 0, keep original width
            video_height: rescaled video height. If 0, keep original height
            video_min_dimension: rescale video so that min(height, width) =
                ``video_min_dimension``. If 0, keep original video resolution.
                Note only one of (``video_width``, ``video_height``)
                and (``video_min_dimension``) can be set
            audio_samples: desired audio sample rate. If 0, keep original
                audio sample rate.
            step_between_clips: Number of frames between each clip.
            frame_rate: desired video frame rate. If None, keep original video
                frame rate.
            clips_per_video: Number of clips to sample from each video
            video_dir: path to video folder
            splits_dir: path to dataset splitting file folder
            fold: UCF101 dataset has 3 folds. Valid values are 1, 2 and 3.
            metadata_filepath: path to the dataset meta data

        """
        # dataset metadata includes the path of video file, the pts of frames in
        # the video and other meta info such as video fps, duration, audio sample rate.
        # Users do not need to know the details of metadata. The computing, loading
        # and saving logic of metata are all handled inside of the dataset.
        # Given the "metadata_file" path, if such file exists, we load it as meta data.
        # Otherwise, we compute the meta data, and save it at "metadata_file" path.
        metadata = None
        if os.path.exists(metadata_filepath):
            metadata = UCF101Dataset.load_metadata(
                metadata_filepath, video_dir=video_dir, update_file_path=True
            )

        dataset = UCF101(
            video_dir,
            splits_dir,
            frames_per_clip,
            step_between_clips=step_between_clips,
            frame_rate=frame_rate,
            _precomputed_metadata=metadata,
            fold=fold,
            train=True if split == "train" else False,
            num_workers=torch.get_num_threads() // 2,  # heuristically use half threads
            _video_width=video_width,
            _video_height=video_height,
            _video_min_dimension=video_min_dimension,
            _audio_samples=audio_samples,
        )
        metadata = dataset.metadata
        if metadata and not os.path.exists(metadata_filepath):
            UCF101Dataset.save_metadata(metadata, metadata_filepath)

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
    def from_config(cls, config: Dict[str, Any]) -> "UCF101Dataset":
        """Instantiates a UCF101Dataset from a configuration.

        Args:
            config: A configuration for a UCF101Dataset.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A UCF101Dataset instance.
        """
        required_args = ["split", "metadata_file", "video_dir", "splits_dir"]
        assert all(
            arg in config for arg in required_args
        ), f"The arguments {required_args} are all required."

        split = config["split"]
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
            config["video_dir"],
            config["splits_dir"],
            config["fold"]
            if "fold" in config
            else 1,  # UCF101 has 3 folds. Use fold 1 by default
            config["metadata_file"],
        )
