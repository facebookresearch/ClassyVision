#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch
from classy_vision.generic.distributed_util import all_reduce_sum
from classy_vision.generic.util import convert_to_one_hot, is_pos_int
from classy_vision.meters import ClassyMeter

from . import register_meter


@register_meter("precision_at_k")
class PrecisionAtKMeter(ClassyMeter):
    """
    Meter to calculate top-k precision for single-label or multi-label
    image classification task. Note, ties are resolved randomly.
    """

    def __init__(self, topk, target_is_one_hot=True, num_classes=-1):
        """
        args:
            topk: list of int `k` values.
            target_is_one_hot: boolean, if class labels are one-hot encoded.
            num_classes: int, number of classes.
        """
        assert isinstance(topk, list), "topk must be a list"
        assert len(topk) > 0, "topk list should have at least one element"
        assert [is_pos_int(x) for x in topk], "each value in topk must be >= 1"
        if not target_is_one_hot:
            assert (
                type(num_classes) == int and num_classes > 0
            ), "num_classes must be positive integer"

        self._topk = topk
        self._target_is_one_hot = target_is_one_hot
        self._num_classes = num_classes

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
    def from_config(cls, config: Dict[str, Any]) -> "PrecisionAtKMeter":
        """Instantiates a PrecisionAtKMeter from a configuration.

        Args:
            config: A configuration for a PrecisionAtKMeter.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A PrecisionAtKMeter instance.
        """
        return cls(
            topk=config["topk"],
            target_is_one_hot=config.get("target_is_one_hot", True),
            num_classes=config.get("num_classes", -1),
        )

    @property
    def name(self):
        return "precision_at_k"

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
            "top_{}".format(k): (correct_predictions[k] / (k * sample_count)).item()
            if sample_count
            else 0.0
            for k in self._topk
        }

    def get_classy_state(self):
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

    def __repr__(self):
        return repr({"name": self.name, "value": self.value})

    def update(self, model_output, target, **kwargs):
        """
        args:
            model_output: tensor of shape (B, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B, C), one-hot encoded
                          or integer encoded or tensor of shape (B),
                          integer encoded.
            Note: For binary classification, C=2.
                  For integer encoded target, C=1.
        """
        target_shape_list = list(target.size())

        if self._target_is_one_hot is False:
            assert len(target_shape_list) == 1 or (
                len(target_shape_list) == 2 and target_shape_list[1] == 1
            ), "Integer encoded target must be single labeled"
            target = convert_to_one_hot(target.view(-1, 1), self._num_classes)

        assert (
            torch.min(target.eq(0) + target.eq(1)) == 1
        ), "Target must be one-hot encoded vector"

        # Due to dummy samples, in some corner cases, the whole batch could
        # be dummy samples, in that case we want to not update meters on that
        # process
        if model_output.shape[0] == 0:
            return
        _, pred_classes = model_output.topk(
            max(self._topk), dim=1, largest=True, sorted=True
        )
        pred_mask_tensor = torch.zeros(target.size())
        for i, k in enumerate(self._topk):
            pred_mask_tensor.zero_()
            self._curr_correct_predictions_k[i] += torch.sum(
                # torch.min is used to simulate AND between binary
                # tensors. If tensors are not binary, this will fail.
                torch.min(
                    pred_mask_tensor.scatter_(1, pred_classes[:, :k], 1.0),
                    target.float(),
                )
            ).item()
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
            precision_meter cannot be larger than num_classes: \
            {}".format(
            model_output_shape[1]
        )
