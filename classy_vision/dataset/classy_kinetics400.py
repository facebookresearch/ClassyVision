#!/usr/bin/env python3

import torch
from torchvision.datasets.kinetics import Kinetics400

from . import register_dataset
from .classy_video_dataset import ClassyVideoDataset
from .core import WrapTorchVisionVideoDataset
from .transforms.util_video import build_video_field_transform_default


@register_dataset("kinetics400")
class Kinetics400Dataset(ClassyVideoDataset):
    """
    Kinetics data in 400 classes.
    Assume videos are already trimmed to 10-second clip.
    """

    def __init__(self, config):
        super(Kinetics400Dataset, self).__init__(config)

        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
            video_width,
            video_height,
            video_min_dimension,
            audio_samples,
            step_between_clips,
            frame_rate,
            precomputed_metadata_filepath,
            save_metadata_filepath,
        ) = self.parse_config(config)

        metadata = None
        if precomputed_metadata_filepath is not None:
            metadata = self.load_metadata(precomputed_metadata_filepath)

        dataset = Kinetics400(
            config["video_dir"],
            config["frames_per_clip"],
            step_between_clips=step_between_clips,
            frame_rate=frame_rate,
            _precomputed_metadata=metadata,
            extensions=config["extensions"],
            num_workers=torch.get_num_threads(),
            _video_width=video_width,
            _video_height=video_height,
            _video_min_dimension=video_min_dimension,
            _audio_samples=audio_samples,
        )
        self.metadata = dataset.metadata
        if self.metadata and save_metadata_filepath:
            self.save_metadata(save_metadata_filepath)

        dataset = WrapTorchVisionVideoDataset(dataset)

        transform = build_video_field_transform_default(transform_config, self._split)
        self.dataset = self.wrap_dataset(
            dataset,
            transform,
            batchsize_per_replica,
            shuffle=shuffle,
            subsample=num_samples,
        )
