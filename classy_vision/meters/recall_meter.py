#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from classy_vision.generic.distributed_util import all_reduce_sum
from classy_vision.generic.util import convert_to_one_hot, is_pos_int
from classy_vision.meters.classy_meter import ClassyMeter

from . import register_meter


@register_meter("recall_at_k")
class RecallAtKMeter(ClassyMeter):
    """Meter to calculate top-k recall for single- or multi-label
       image classification task.
    """

    def __init__(self, topk, target_is_one_hot=True, num_classes=None):
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
        self._state_unsynced = False

        # _total_* variables store running, in-sync totals for the
        # metrics. These should not be communicated / summed.
        self._total_correct_predictions_k = None
        self._total_correct_targets = None

        # _curr_* variables store counts since the last sync. Only
        # these should be summed across workers and they are reset
        # after each communication
        self._curr_correct_predictions_k = None
        self._curr_correct_targets = None

        # Initialize all values properly
        self.reset()

    @classmethod
    def from_config(cls, config):
        return cls(
            topk=config["topk"],
            target_is_one_hot=config.get("target_is_one_hot", True),
            num_classes=config.get("num_classes", None),
        )

    @property
    def name(self):
        return "recall_at_k"

    def _sync_state(self):
        # Communications
        self._curr_correct_predictions_k = all_reduce_sum(
            self._curr_correct_predictions_k
        )
        self._curr_correct_targets = all_reduce_sum(self._curr_correct_targets)

        # Store results
        self._total_correct_predictions_k += self._curr_correct_predictions_k
        self._total_correct_targets += self._curr_correct_targets

        # Reset values until next sync
        self._curr_correct_predictions_k.zero_()
        self._curr_correct_targets.zero_()

    @property
    def value(self):
        if self._state_unsynced:
            self._sync_state()
            self._state_unsynced = False
        return {
            "top_{}".format(k): (
                correct_prediction_k.item() / self._total_correct_targets.item()
            )
            if self._total_correct_targets
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
            "total_correct_targets": self._total_correct_targets.clone(),
            "curr_correct_targets": self._curr_correct_targets.clone(),
            "curr_correct_predictions_k": self._curr_correct_predictions_k.clone(),
        }

    @meter_state_dict.setter
    def meter_state_dict(self, state):
        assert self._topk == state["top_k"], "Incompatible top-k for recall!"

        # Restore the state -- correct_predictions and correct_targets.
        self.reset()
        self._total_correct_predictions_k = state["total_correct_predictions"].clone()
        self._total_correct_targets = state["total_correct_targets"].clone()
        self._curr_correct_predictions_k = state["curr_correct_predictions_k"].clone()
        self._curr_correct_targets = state["curr_correct_targets"].clone()
        self._state_unsynced = True

    def __repr__(self):
        return repr({"name": self.name, "value": self.value})

    def update(self, model_output, target):
        """
        args:
            model_output: tensor of shape (B, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B, C), one-hot encoded
                          or integer encoded.

            Note: For binary classification, C=2.
                  For integer encoded target, C=1.
        """

        if self._target_is_one_hot is False:
            assert target.shape[1] == 1, "Integer encoded target must be single labeled"
            target = convert_to_one_hot(target, self._num_classes)

        assert (
            torch.min(target.eq(0) + target.eq(1)) == 1
        ), "Target must be one-hot encoded vector"
        self._state_unsynced = True
        # Due to dummy samples, in some corner cases, the whole batch could
        # be dummy samples, in that case we want to not update meters on that
        # process but still want to update the state to unsynced as the state
        # should be same for all processes.
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
        self._curr_correct_targets += target.sum().item()

    def reset(self):
        self._total_correct_predictions_k = torch.zeros(len(self._topk))
        self._total_correct_targets = torch.zeros(1)
        self._curr_correct_predictions_k = torch.zeros(len(self._topk))
        self._curr_correct_targets = torch.zeros(1)

    def validate(self, model_output_shape, target_shape):
        assert (
            len(model_output_shape) == 2
        ), "model_output_shape must be (B, C) \
            Found shape {}".format(
            model_output_shape
        )
        assert (
            len(target_shape) == 2
        ), "target_shape must be (B, C) \
            Found shape {}".format(
            target_shape
        )
        assert (
            max(self._topk) < model_output_shape[1]
        ), "k in top_k, for \
            recall_meter cannot be larger than num_classes: \
            {}".format(
            model_output_shape[1]
        )
