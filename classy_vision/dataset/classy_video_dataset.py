#!/usr/bin/env python3


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

        step_between_clips = (
            config["step_between_clips"] if "step_between_clips" in config else 1
        )
        frame_rate = config["frame_rate"] if "frame_rate" in config else None
        precomputed_metadata_filepath = (
            config["load_metadata_file"] if "load_metadata_file" in config else None
        )
        save_metadata_filepath = (
            config["save_metadata_file"] if "save_metadata_file" in config else None
        )

        (transform_config, batchsize_per_replica, shuffle, num_samples) = super(
            ClassyVideoDataset, self
        ).parse_config(config)

        return (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
            step_between_clips,
            frame_rate,
            precomputed_metadata_filepath,
            save_metadata_filepath,
        )
