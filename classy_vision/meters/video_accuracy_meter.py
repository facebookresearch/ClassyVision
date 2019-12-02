#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch
from classy_vision.generic.util import is_pos_int
from classy_vision.meters import ClassyMeter
from classy_vision.meters.accuracy_meter import AccuracyMeter

from . import register_meter


@register_meter("video_accuracy")
class VideoAccuracyMeter(ClassyMeter):
    """Meter to calculate top-k video-level accuracy for single label
       video classification task. Video-level accuarcy is computed by averaging
       clip-level predictions and compare the reslt with video-level groundtruth
       label.
    """

    def __init__(self, topk, clips_per_video_train, clips_per_video_test):
        """
        args:
            topk: list of int `k` values.
            clips_per_video_train: No. of clips sampled per video at train time
            clips_per_video_test: No. of clips sampled per video at test time
        """
        assert isinstance(topk, list), "topk must be a list"
        assert len(topk) > 0, "topk list should have at least one element"
        assert [is_pos_int(x) for x in topk], "each value in topk must be >= 1"

        self._clips_per_video_train = clips_per_video_train
        self._clips_per_video_test = clips_per_video_test
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
    def value(self):
        return self._accuracy_meter.value

    def sync_state(self):
        self._accuracy_meter.sync_state()

    def get_classy_state(self):
        """Contains the states of the meter.
        """
        state = {}
        state["accuracy_state"] = self._accuracy_meter.get_classy_state()
        state["name"] = "video_accuracy"
        state["clips_per_video_train"] = self._clips_per_video_train
        state["clips_per_video_test"] = self._clips_per_video_test
        return state

    def set_classy_state(self, state):
        assert (
            "video_accuracy" == state["name"]
        ), "State name {state_name} does not match meter name {obj_name}".format(
            state_name=state["name"], obj_name=self.name
        )
        assert (
            self._clips_per_video_train == state["clips_per_video_train"]
        ), "incompatible clips_per_video_train for video accuracy"
        assert (
            self._clips_per_video_test == state["clips_per_video_test"]
        ), "incompatible clips_per_video_test for video accuracy"
        # Restore the state -- correct_predictions and sample_count.
        self.reset()
        self._accuracy_meter.set_classy_state(state["accuracy_state"])

    def __repr__(self):
        return repr({"name": self.name, "value": self._accuracy_meter.value})

    def update(self, model_output, target, is_train, **kwargs):
        """
        args:
            model_output: tensor of shape (B * clips_per_video, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B * clips_per_video).
            is_train     if True, it is training stage when meter is updated

            Note: For binary classification, C=2.
        """
        num_clips = len(model_output)
        if num_clips == 0:
            # It is possible that a minibatch entirely contains dummy samples
            # when dataset is sharded. In such case, the effective target and output
            # can be empty, and we immediately return
            return

        clips_per_video = (
            self._clips_per_video_train if is_train else self._clips_per_video_test
        )
        assert num_clips % clips_per_video == 0, (
            "For video model testing, batch size must be a multplier of No. of "
            "clips per video"
        )
        num_videos = num_clips // clips_per_video
        for i in range(num_videos):
            clip_labels = target[i * clips_per_video : (i + 1) * clips_per_video]
            assert (
                len(torch.unique(clip_labels)) == 1
            ), "all clips from the same video should have same label"

        video_target = target[::clips_per_video]
        video_model_output = torch.mean(
            torch.reshape(model_output, (num_videos, clips_per_video, -1)), 1
        )
        self._accuracy_meter.update(video_model_output, video_target)

    def reset(self):
        self._accuracy_meter.reset()

    def validate(self, model_output_shape, target_shape):
        self._accuracy_meter.validate(model_output_shape, target_shape)
