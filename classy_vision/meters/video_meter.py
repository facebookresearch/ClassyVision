#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from classy_vision.meters import ClassyMeter


class VideoMeter(ClassyMeter):
    """An abstraction of meter for evaluating video models.

    Video-level metric is computed by averaging clip-level predictions and
    compare the result with video-level groundtruth label.

    This meter abstraction can wrap conventional classy meters by passing
    averaged clip-level predictions to the meter needed for video level metrics.
    """

    def __init__(self, clips_per_video_train, clips_per_video_test):
        """Constructor of VideoMeter class.

        Args:
            clips_per_video_train: No. of clips sampled per video at train time
            clips_per_video_test: No. of clips sampled per video at test time
        """

        self._clips_per_video_train = clips_per_video_train
        self._clips_per_video_test = clips_per_video_test

    @property
    def value(self):
        return self.meter.value

    def sync_state(self):
        self.meter.sync_state()

    @property
    def meter(self) -> "ClassyMeter":
        """Every video meter should implement to have its own internal meter.

        It consumes the video level predictions and ground truth label, and compute
        the actual metrics.

        Returns:
            An instance of ClassyMeter.
        """
        raise NotImplementedError

    def get_classy_state(self):
        """Contains the states of the meter.
        """
        state = {}
        state["meter_state"] = self.meter.get_classy_state()
        state["name"] = self.name
        state["clips_per_video_train"] = self._clips_per_video_train
        state["clips_per_video_test"] = self._clips_per_video_test
        return state

    def set_classy_state(self, state):
        assert (
            self.name == state["name"]
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
        self.meter.set_classy_state(state["meter_state"])

    def __repr__(self):
        return repr({"name": self.name, "value": self.value})

    def update(self, model_output, target, is_train, **kwargs):
        """Updates any internal state of meter with new model output and target.

        Args:
            model_output: tensor of shape (B * clips_per_video, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B * clips_per_video).
            is_train     if True, it is training stage when meter is updated

            Note: For binary classification, C=2.
        """
        num_clips = len(model_output)
        clips_per_video = (
            self._clips_per_video_train if is_train else self._clips_per_video_test
        )

        if not num_clips % clips_per_video == 0:
            logging.info(
                "Skip meter update. Because for video model testing, batch size "
                "is expected to be a multplier of No. of clips per video. "
                "num_clips: %d, clips_per_video: %d" % (num_clips, clips_per_video)
            )
            return

        num_videos = num_clips // clips_per_video
        for i in range(num_videos):
            clip_labels = target[i * clips_per_video : (i + 1) * clips_per_video]
            if clip_labels.ndim == 1:
                # single label
                assert (
                    len(torch.unique(clip_labels)) == 1
                ), "all clips from the same video should have same label"
            elif clip_labels.ndim == 2:
                # multi-hot label
                for j in range(1, clip_labels.shape[0]):
                    assert torch.equal(
                        clip_labels[0], clip_labels[j]
                    ), "all clips from the same video should have the same labels"
            else:
                raise ValueError(
                    "dimension of clip label matrix should be either 1 or 2"
                )

        video_model_output = torch.mean(
            torch.reshape(model_output, (num_videos, clips_per_video, -1)), 1
        )
        video_target = target[::clips_per_video]
        self.meter.update(video_model_output, video_target)

    def reset(self):
        self.meter.reset()

    def validate(self, model_output_shape, target_shape):
        self.meter.validate(model_output_shape, target_shape)
