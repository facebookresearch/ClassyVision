#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from classy_vision.generic.util import is_pos_int
from classy_vision.meters.accuracy_meter import AccuracyMeter

from . import register_meter
from .video_meter import VideoMeter


@register_meter("video_accuracy")
class VideoAccuracyMeter(VideoMeter):
    """Meter to calculate top-k video-level accuracy for single/multi label
       video classification task.

    Video-level accuarcy is computed by averaging clip-level predictions and
    compare the reslt with video-level groundtruth label.
    """

    def __init__(self, topk, clips_per_video_train, clips_per_video_test):
        """
        Args:
            topk: list of int `k` values.
            clips_per_video_train: No. of clips sampled per video at train time
            clips_per_video_test: No. of clips sampled per video at test time
        """
        super().__init__(clips_per_video_train, clips_per_video_test)
        assert isinstance(topk, list), "topk must be a list"
        assert len(topk) > 0, "topk list should have at least one element"
        assert [is_pos_int(x) for x in topk], "each value in topk must be >= 1"

        self._accuracy_meter = AccuracyMeter(topk)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoAccuracyMeter":
        """Instantiates a VideoAccuracyMeter from a configuration.

        Args:
            config: A configuration for a VideoAccuracyMeter.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A VideoAccuracyMeter instance.
        """
        return cls(
            topk=config["topk"],
            clips_per_video_train=config.get("clips_per_video_train", 1),
            clips_per_video_test=config["clips_per_video_test"],
        )

    @property
    def name(self):
        return "video_accuracy"

    @property
    def meter(self):
        return self._accuracy_meter
