#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Callable, Dict, Optional

import torch
from classy_vision.generic.distributed_util import get_rank, get_world_size
from torch.utils.data import Sampler
from torchvision import get_video_backend
from torchvision.datasets.samplers.clip_sampler import (
    DistributedSampler,
    RandomClipSampler,
    UniformClipSampler,
)

from .classy_dataset import ClassyDataset


class MaxLengthClipSampler(Sampler):
    """MaxLengthClipSampler is a thin wrapper on top of clip samplers in TorchVision.

    It takes as input a TorchVision clip sampler, and an optional argument
    `num_samples` to limit the number of samples.
    """

    def __init__(self, clip_sampler, num_samples=None):
        """The constructor method of MaxLengthClipSampler.

        Args:
            clip_sampler: clip sampler without a limit on the total number of clips
                it can sample, such as RandomClipSampler and UniformClipSampler.
            num_samples: if provided, it denotes the maximal number of clips the sampler
                will return

        """
        self.clip_sampler = clip_sampler
        self.num_samples = num_samples

    def __iter__(self):
        num_samples = len(self)
        n = 0
        for clip in self.clip_sampler:
            if n < num_samples:
                yield clip
                n += 1
            else:
                break

    def __len__(self):
        full_size = len(self.clip_sampler)
        if self.num_samples is None:
            return full_size

        return min(full_size, self.num_samples)


class ClassyVideoDataset(ClassyDataset):
    """Interface specifying what a ClassyVision video dataset is expected to provide.

    This dataset considers every video as a collection of video clips of fixed size,
    specified by ``frames_per_clip``, where the step in frames between each clip
    is given by ``step_between_clips``. It uses a clip sampler to sample
    a specified number of clips (``clips_per_video``) from each video.
    For training set, a random clip sampler is used to
    sample a small number of clips (e.g. 1) from each video
    For testing set, a uniform clip sampler is used to evenly sample a large
    number of clips (e.g. 10) from the video.

    To give an example, for 2 videos with 10 and 15 frames respectively,
    if ``frames_per_clip=5`` and ``step_between_clips=5``, the dataset size
    will be (2 + 3) = 5, where the first two elements will come from video 1,
    and the next three elements from video 2. Note that we drop clips which do
    not have exactly ``frames_per_clip`` elements, so not all frames in a video
    may be present.
    """

    def __init__(
        self,
        dataset: Any,
        split: str,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Callable,
        num_samples: Optional[int],
        clips_per_video: int,
    ):
        """The constructor method of ClassyVideoDataset.

        Args:
            dataset: the underlying video dataset from either TorchVision or other
                source. It should have an attribute *video_clips* of type
                `torchvision.datasets.video_utils.VideoClips <https://github.com/
                pytorch/vision/blob/master/torchvision/datasets/
                video_utils.py#L46/>`_
            split: dataset split. Must be either "train" or "test"
            batchsize_per_replica: batch size per model replica
            shuffle: If true, shuffle video clips.
            transform: callable function to transform video clip sample from
                ClassyVideoDataset
            num_samples: If provided, return at most `num_samples` video clips
            clips_per_video: The number of clips sampled from each video

        """
        super(ClassyVideoDataset, self).__init__(
            dataset, batchsize_per_replica, shuffle, transform, num_samples
        )
        # Assignments:
        self.clips_per_video = clips_per_video
        self.split = split
        self.video_backend = get_video_backend()

    @classmethod
    def parse_config(cls, config: Dict[str, Any]):
        """Parse config to prepare arguments needed by the class constructor."""
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
    def load_metadata(
        cls,
        filepath: str,
        video_dir: Optional[str] = None,
        update_file_path: bool = False,
    ) -> Dict[str, Any]:
        """Load pre-computed video dataset meta data.

        Video dataset meta data computation takes minutes on small dataset and hours
        on large dataset, and thus is time-consuming. However, it only needs to be
        computed once, and can be saved into a file via :func:`save_metadata`.

        The format of meta data is defined in `TorchVision <https://github.com/
        pytorch/vision/blob/master/torchvision/datasets/video_utils.py#L131/>`_.

        For each video, meta data contains the video file path, presentation
        timestamps of all video frames, and video fps.

        Args:
            filepath: file path of pre-computed meta data
            video_dir: If provided, the folder where video files are stored.
            update_file_path: If true, replace the directory part of video file path
                in meta data with the actual video directory provided in `video_dir`.
                This is necessary for successsfully reusing pre-computed meta data
                when video directory has been moved and is no longer consitent
                with the full video file path saved in the meta data.
        """
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
    def save_metadata(cls, metadata: Dict[str, Any], filepath: str):
        """Save dataset meta data into a file.

        Args:
            metadata: dataset meta data, which contains video meta infomration, such
                as video file path, video fps, video frame timestamp in each video.
                For the format of dataset meta data, check the `TorchVision
                documentation <https://github.com/pytorch/vision/blob/master/
                torchvision/datasets/video_utils.py#L132-L137/>`_.
            filepath: file path where the meta data will be saved

        """
        filedir = os.path.dirname(filepath)
        if len(filedir) > 0:
            # When filepath includes the absoluate/relative directory, we create the
            # directory if it does not exist yet
            try:
                os.makedirs(filedir, exist_ok=True)
                logging.info(f"Save metadata to file: {filedir}")
            except Exception as err:
                logging.warn(f"Fail to create folder: {filedir}")
                raise err

        try:
            torch.save(metadata, filepath)
        except ValueError:
            logging.warn(f"Fail to save metadata to file: {filepath}")

    @property
    def video_clips(self):
        """Attribute video_clips.

        It is used in ``_get_sampler`` method. Its data type should be
            `torchvision.datasets.video_utils.VideoClips <https://github.com/
            pytorch/vision/blob/master/torchvision/datasets/video_utils.py#L46/>`_.
        """
        return self.dataset.video_clips

    def _get_sampler(self, epoch) -> "DistributedSampler":
        if self.split == "train":
            # For video model training, we don't necessarily want to use all possible
            # clips in the video in one training epoch. More often, we randomly
            # sample at most N clips per training video. In practice, N is often 1
            clip_sampler = RandomClipSampler(self.video_clips, self.clips_per_video)
        else:
            # For video model testing, we sample N evenly spaced clips per test
            # video. We will simply average predictions over them
            clip_sampler = UniformClipSampler(self.video_clips, self.clips_per_video)
        clip_sampler = MaxLengthClipSampler(clip_sampler, num_samples=self.num_samples)
        world_size = get_world_size()
        rank = get_rank()
        sampler = DistributedSampler(
            clip_sampler,
            num_replicas=world_size,
            rank=rank,
            shuffle=self.shuffle,
            group_size=self.clips_per_video,
        )
        sampler.set_epoch(epoch)
        return sampler

    def _worker_init_fn(self, worker_id):
        # we need to set video backend in the worker process explicitly
        # because the global variable `_video_backend` in TorchVision will
        # always start with the default value `pyav` when multiprocessing
        # context other than `fork` is used, and it won't inherit the value of
        # `_video_backend` in the main process

        from torchvision import set_video_backend

        set_video_backend(self.video_backend)

    def iterator(self, *args, **kwargs):
        kwargs["worker_init_fn"] = self._worker_init_fn
        return super(ClassyVideoDataset, self).iterator(*args, **kwargs)
