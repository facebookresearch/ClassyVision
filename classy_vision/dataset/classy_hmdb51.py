#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torchvision.datasets.hmdb51 import HMDB51

from . import register_dataset
from .classy_video_dataset import ClassyVideoDataset
from .transforms.util_video import build_video_field_transform_default


@register_dataset("hmdb51")
class HMDB51Dataset(ClassyVideoDataset):
    """
    HMDB51 is an action recognition video dataset, and it has 51 classes.
    <http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>

    This dataset consider every video as a collection of video clips of fixed size,
    specified by ``frames_per_clip``, where the step in frames between each clip
    is given by ``step_between_clips``. It uses clip sampler to sample clips
    from each video. For training set, a random clip sampler is used to
    sample a small number of clips (e.g. 1) from each video
    For testing set, a uniform clip sampler is used to evenly sample a large
    number of clips (e.g. 10) from the video.

    To give an example, for 2 videos with 10 and 15 frames respectively,
    if ``frames_per_clip=5`` and ``step_between_clips=5``, the dataset size
    will be (2 + 3) = 5, where the first two elements will come from video 1,
    and the next three elements from video 2. Note that we drop clips which do
    not have exactly ``frames_per_clip`` elements, so not all frames in a video
    might be present.

    It is built on top of HMDB51 dataset class in TorchVision.
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
        video_dir,
        splits_dir,
        fold,
        metadata_filepath,
    ):
        """
        Args:
            split (str): dataset split which can be either "train" or "test"
            shuffle (bool): If true, shuffle the dataset
            transform (dict): a dict where transforms video and audio data
            num_samples (optional(int)): if not None, it will subsample dataset
            frames_per_clip (int): the No. of frames in a video clip
            video_width (int): rescaled video width. If 0, keep original width
            video_height (int): rescaled video height. If 0, keep original height
            video_min_dimension (int): rescale video so that min(height, width) =
                video_min_dimension. If 0, keep original video resolution. Note
                only one of (video_width, video_height) and (video_min_dimension)
                can be set
            audio_samples (int): desired audio sample rate. If 0, keep original
                audio sample rate.
            step_between_clips (int): No. of frames between each clip.
            frame_rate (optional(int)): desired video frame rate. If None, keep
                orignal video frame rate.
            clips_per_video (int): No. of clips to sample from each video
            video_dir (str): path to video folder
            splits_dir (str): path to dataset splitting file folder
            fold (int): HMDB51 dataset has 3 folds. Valid values are 1, 2 and 3.
            metadata_filepath (str): path to the dataset meta data
        """
        # dataset metadata includes the path of video file, the pts of frames in
        # the video and other meta info such as video fps, duration, audio sample rate.
        # Users do not need to know the details of metadata. The computing, loading
        # and saving logic of metata are all handled inside of the dataset.
        # Given the "metadata_file" path, if such file exists, we load it as meta data.
        # Otherwise, we compute the meta data, and save it at "metadata_file" path.
        metadata = None
        if os.path.exists(metadata_filepath):
            metadata = HMDB51Dataset.load_metadata(
                metadata_filepath, video_dir=video_dir, update_file_path=True
            )

        dataset = HMDB51(
            video_dir,
            splits_dir,
            frames_per_clip,
            step_between_clips=step_between_clips,
            frame_rate=frame_rate,
            _precomputed_metadata=metadata,
            fold=fold,
            train=(split == "train"),
            num_workers=torch.get_num_threads(),
            _video_width=video_width,
            _video_height=video_height,
            _video_min_dimension=video_min_dimension,
            _audio_samples=audio_samples,
        )
        metadata = dataset.metadata
        if metadata and not os.path.exists(metadata_filepath):
            HMDB51Dataset.save_metadata(metadata, metadata_filepath)

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
    def from_config(cls, config):
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
            else 1,  # HMDB51 has 3 folds. Use fold 1 by default
            config["metadata_file"],
        )
