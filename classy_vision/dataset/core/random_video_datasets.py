#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ...generic.util import torch_seed


class RandomVideoDataset:
    def __init__(
        self,
        num_classes,
        split,
        num_samples,
        frames_per_clip,
        video_width,
        video_height,
        audio_samples,
        clips_per_video,
        seed=10,
    ):
        self.num_classes = num_classes
        self.split = split
        # video config
        self.video_channels = 3
        self.num_samples = num_samples
        self.frames_per_clip = frames_per_clip
        self.video_width = video_width
        self.video_height = video_height
        # audio config
        self.audio_samples = audio_samples
        self.clips_per_video = clips_per_video
        # misc config
        self.seed = seed

    def __getitem__(self, idx):
        if self.split == "train":
            # assume we only sample 1 clip from each training video
            target_seed_offset = idx
        else:
            # for video model testing, clips from the same video share the same
            # target label
            target_seed_offset = idx // self.clips_per_video
        with torch_seed(self.seed + target_seed_offset):
            target = torch.randint(0, self.num_classes, (1,)).item()

        with torch_seed(self.seed + idx):
            return {
                "input": {
                    "video": torch.randint(
                        0,
                        256,
                        (
                            self.frames_per_clip,
                            self.video_height,
                            self.video_width,
                            self.video_channels,
                        ),
                        dtype=torch.uint8,
                    ),
                    "audio": torch.rand((self.audio_samples, 1), dtype=torch.float),
                },
                "target": target,
            }

    def __len__(self):
        return self.num_samples
