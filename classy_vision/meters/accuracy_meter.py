#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from classy_vision.generic.distributed_util import all_reduce_sum
from classy_vision.generic.util import is_pos_int
from classy_vision.meters.classy_meter import ClassyMeter

from . import register_meter


@register_meter("accuracy")
class AccuracyMeter(ClassyMeter):
    """Meter to calculate top-k accuracy for single label
       image classification task.
    """

    def __init__(self, topk):
        """
        args:
            topk: list of int `k` values.
        """
        assert isinstance(topk, list), "topk must be a list"
        assert len(topk) > 0, "topk list should have at least one element"
        assert [is_pos_int(x) for x in topk], "each value in topk must be >= 1"
        self._topk = topk
        self._state_unsynced = False

        # _total_* variables store running, in-sync totals for the
        # metrics. These should not be communicated / summed.
        self._total_correct_predictions_k = None
        self._total_sample_count = None

        # _curr_* variables store counts since the last sync. Only
        # these should be summed across workers and they are reset
        # after each communication
        self._curr_correct_predictions_k = None
        self._curr_sample_count = None

        # Initialize all values properly
        self.reset()

    @property
    def name(self):
        return "accuracy"

    def _sync_state(self):
        # Communications
        self._curr_correct_predictions_k = all_reduce_sum(
            self._curr_correct_predictions_k
        )
        self._curr_sample_count = all_reduce_sum(self._curr_sample_count)

        # Store results
        self._total_correct_predictions_k += self._curr_correct_predictions_k
        self._total_sample_count += self._curr_sample_count

        # Reset values until next sync
        self._curr_correct_predictions_k.zero_()
        self._curr_sample_count.zero_()

    @property
    def value(self):
        if self._state_unsynced:
            self._sync_state()
            self._state_unsynced = False
        return {
            "top_{}".format(k): (correct_prediction_k / self._total_sample_count).item()
            if self._total_sample_count
            else 0.0
            for k, correct_prediction_k in zip(
                self._topk, self._total_correct_predictions_k
            )
        }

    @property
    def meter_state_dict(self):
        """Contains the states of the meter.
        """
        return {
            "name": self.name,
            "top_k": self._topk,
            "total_correct_predictions": self._total_correct_predictions_k.clone(),
            "total_sample_count": self._total_sample_count.clone(),
            "curr_sample_count": self._curr_sample_count.clone(),
            "curr_correct_predictions_k": self._curr_correct_predictions_k.clone(),
        }

    @meter_state_dict.setter
    def meter_state_dict(self, state):
        assert self._topk == state["top_k"], "Incompatible top-k for accuracy!"

        # Restore the state -- correct_predictions and sample_count.
        self.reset()
        self._total_correct_predictions_k = state["total_correct_predictions"].clone()
        self._total_sample_count = state["total_sample_count"].clone()
        self._curr_correct_predictions_k = state["curr_correct_predictions_k"].clone()
        self._curr_sample_count = state["curr_sample_count"].clone()
        self._state_unsynced = True

    def __repr__(self):
        return repr({"name": self.name, "value": self.value})

    def update(self, model_output, target):
        """
        args:
            model_output: tensor of shape (B, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B).
            Note: For binary classification, C=2.
        """
        self._state_unsynced = True
        # Due to dummy samples, in some corner cases, the whole batch could
        # be dummy samples, in that case we want to not update meters on that
        # process but still want to update the state to unsynced as the state
        # should be same for all processes.
        if model_output.shape[0] == 0:
            return
        _, pred = model_output.topk(max(self._topk), dim=1, largest=True, sorted=True)

        correct_predictions = pred.eq(target.unsqueeze(1).expand_as(pred))
        for i, k in enumerate(self._topk):
            self._curr_correct_predictions_k[i] += (
                correct_predictions[:, :k].float().sum().item()
            )
        self._curr_sample_count += model_output.shape[0]

    def reset(self):
        self._total_correct_predictions_k = torch.zeros(len(self._topk))
        self._total_sample_count = torch.zeros(1)
        self._curr_correct_predictions_k = torch.zeros(len(self._topk))
        self._curr_sample_count = torch.zeros(1)

    def validate(self, model_output_shape, target_shape):
        assert (
            len(model_output_shape) == 2
        ), "model_output_shape must be (B, C) \
            Found shape {}".format(
            model_output_shape
        )
        assert (
            len(target_shape) == 1
        ), "target_shape must be (B) \
            Found shape {}".format(
            target_shape
        )
        assert (
            max(self._topk) < model_output_shape[1]
        ), "k in top_k, for \
            accuracy_meter cannot be larger than num_classes: \
            {}".format(
            model_output_shape[1]
        )
