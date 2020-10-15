#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch
from classy_vision.generic.distributed_util import all_reduce_sum
from classy_vision.generic.util import is_pos_int, maybe_convert_to_one_hot
from classy_vision.meters import ClassyMeter

from . import register_meter


@register_meter("accuracy")
class AccuracyMeter(ClassyMeter):
    """Meter to calculate top-k accuracy for single label/ multi label
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AccuracyMeter":
        """Instantiates a AccuracyMeter from a configuration.

        Args:
            config: A configuration for a AccuracyMeter.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A AccuracyMeter instance.
        """
        return cls(topk=config["topk"])

    @property
    def name(self):
        return "accuracy"

    def sync_state(self):
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
        # Return value based on the local state of meter which
        # includes the local sample count since last sync and the total global sample
        # count obtained at the last sync
        correct_predictions = {
            k: self._curr_correct_predictions_k[i]
            + self._total_correct_predictions_k[i]
            for i, k in enumerate(self._topk)
        }
        sample_count = self._total_sample_count + self._curr_sample_count
        return {
            "top_{}".format(k): (correct_predictions[k] / sample_count).item()
            if sample_count
            else 0.0
            for k in self._topk
        }

    def get_classy_state(self):
        """Contains the states of the meter."""
        return {
            "name": self.name,
            "top_k": self._topk,
            "total_correct_predictions": self._total_correct_predictions_k.clone(),
            "total_sample_count": self._total_sample_count.clone(),
            "curr_sample_count": self._curr_sample_count.clone(),
            "curr_correct_predictions_k": self._curr_correct_predictions_k.clone(),
        }

    def set_classy_state(self, state):
        assert (
            self.name == state["name"]
        ), "State name {state_name} does not match meter name {obj_name}".format(
            state_name=state["name"], obj_name=self.name
        )
        assert (
            self._topk == state["top_k"]
        ), "top-k of state {state_k} does not match object's top-k {obj_k}".format(
            state_k=state["top_k"], obj_k=self._topk
        )

        # Restore the state -- correct_predictions and sample_count.
        self.reset()
        self._total_correct_predictions_k = state["total_correct_predictions"].clone()
        self._total_sample_count = state["total_sample_count"].clone()
        self._curr_correct_predictions_k = state["curr_correct_predictions_k"].clone()
        self._curr_sample_count = state["curr_sample_count"].clone()

    def update(self, model_output, target, **kwargs):
        """
        args:
            model_output: tensor of shape (B, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B, C), which is one-hot /
                          multi-label encoded, or tensor of shape (B) /
                          (B, 1), integer encoded
        """
        # Convert target to 0/1 encoding if isn't
        target = maybe_convert_to_one_hot(target, model_output)

        _, pred = model_output.topk(max(self._topk), dim=1, largest=True, sorted=True)
        for i, k in enumerate(self._topk):
            self._curr_correct_predictions_k[i] += (
                torch.gather(target, dim=1, index=pred[:, :k])
                .max(dim=1)
                .values.sum()
                .item()
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
            len(target_shape) > 0 and len(target_shape) < 3
        ), "target_shape must be (B) or (B, C) \
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
