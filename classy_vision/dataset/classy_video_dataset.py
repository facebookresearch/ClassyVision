#!/usr/bin/env python3
import logging
import os

import torch

from .classy_dataset import ClassyDataset


class ClassyVideoDataset(ClassyDataset):
    """
    Interface specifying what a Classy Vision video dataset can be expected to provide.
    """

    def __init__(self, config):
        super(ClassyVideoDataset, self).__init__(config)

    def parse_config(self, config):
        for config_key in ["frames_per_clip", "video_dir"]:
            assert config_key in config, "%s must be set" % config_key

        video_width = config.get("video_width", 0)
        video_height = config.get("video_height", 0)
        video_min_dimension = config.get("video_min_dimension", 0)
        audio_samples = config.get("audio_samples", 0)
        step_between_clips = config.get("step_between_clips", 1)
        frame_rate = config.get("frame_rate", None)
        precomputed_metadata_filepath = config.get("load_metadata_file", None)
        save_metadata_filepath = config.get("save_metadata_file", None)

        (transform_config, batchsize_per_replica, shuffle, num_samples) = super(
            ClassyVideoDataset, self
        ).parse_config(config)

        return (
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
        )

    def load_metadata(self, filepath):
        assert os.path.exists(filepath), "File not found: %s" % filepath
        metadata = torch.load(filepath)
        return metadata

    def save_metadata(self, filepath):
        filedir = os.path.dirname(filepath)
        if not os.path.exists(filedir):
            try:
                os.mkdirs(filedir)
            except Exception:
                logging.warn("Fail to create folder: %s" % filedir)
                return
        logging.info("Save metadata to file: %s" % filepath)
        try:
            torch.save(self.metadata, filepath)
        except ValueError:
            logging.warn("Fail to save metadata to file: %s" % filepath)
