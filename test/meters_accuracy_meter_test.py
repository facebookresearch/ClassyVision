#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from test.generic.meter_test_utils import ClassificationMeterTest

import torch
from classy_vision import meters
from classy_vision.meters import AccuracyMeter


class TestAccuracyMeter(ClassificationMeterTest):
    def test_accuracy_meter_registry(self):
        accuracy_meter = meters.build_meter({"name": "accuracy", "topk": [1, 2]})
        self.assertTrue(isinstance(accuracy_meter, AccuracyMeter))

    def test_single_meter_update_and_reset(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update.
        """
        meter = AccuracyMeter(topk=[1, 2])

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_output = torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]])

        # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
        target = torch.tensor([0, 1, 2])

        # Only the first sample has top class correct, first and third
        # sample have correct class in top 2
        expected_value = {"top_1": 1 / 3.0, "top_2": 2 / 3.0}

        self.meter_update_and_reset_test(meter, model_output, target, expected_value)

    def test_double_meter_update_and_reset(self):
        meter = AccuracyMeter(topk=[1, 2])

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score...two batches in this test
        model_outputs = [
            torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),
            torch.tensor([[3, 2, 1], [1, 3, 2], [1, 3, 2]]),
        ]

        # Class 0 is the correct class for sample 1, class 2 for
        # sample 2, etc, in both batches
        targets = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])]

        # First batch has top-1 accuracy of 1/3.0, top-2 accuracy of 2/3.0
        # Second batch has top-1 accuracy of 2/3.0, top-2 accuracy of 3/3.0
        expected_value = {"top_1": 3 / 6.0, "top_2": 5 / 6.0}

        self.meter_update_and_reset_test(meter, model_outputs, targets, expected_value)

    def test_meter_invalid_model_output(self):
        meter = AccuracyMeter(topk=[1, 2])
        # This model output has 3 dimensions instead of expected 2
        model_output = torch.tensor(
            [[[3, 2, 1], [1, 2, 3]], [[-1, -3, -4], [-10, -90, -100]]]
        )
        target = torch.tensor([0, 1, 2])

        self.meter_invalid_meter_input_test(meter, model_output, target)

    def test_meter_invalid_target(self):
        meter = AccuracyMeter(topk=[1, 2])
        model_output = torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]])
        # Target has 2 dimensions instead of expected 1
        target = torch.tensor([[0, 1, 2], [0, 1, 2]])

        self.meter_invalid_meter_input_test(meter, model_output, target)

    def test_meter_invalid_topk(self):
        meter = AccuracyMeter(topk=[1, 5])
        model_output = torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]])
        target = torch.tensor([0, 1, 2])

        self.meter_invalid_meter_input_test(meter, model_output, target)

    def test_meter_get_set_classy_state_test(self):
        # In this test we update meter0 with model_output0 & target0
        # and we update meter1 with model_output1 & target1 then
        # transfer the state from meter1 to meter0 and validate they
        # give same expected value.
        # Expected value is the expected value of meter1
        meters = [AccuracyMeter(topk=[1, 2]), AccuracyMeter(topk=[1, 2])]

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_outputs = [
            torch.tensor([[1, 2, 3], [1, 2, 3], [2, 3, 1]]),
            torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),
        ]

        # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
        targets = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])]

        # Value for second update
        expected_value = {"top_1": 1 / 3.0, "top_2": 2 / 3.0}

        self.meter_get_set_classy_state_test(
            meters, model_outputs, targets, expected_value
        )

    def test_meter_distributed(self):
        # Meter0 will execute on one process, Meter1 on the other
        meters = [AccuracyMeter(topk=[1, 2]), AccuracyMeter(topk=[1, 2])]

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_outputs = [
            torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),  # Meter 0
            torch.tensor([[3, 2, 1], [1, 3, 2], [1, 3, 2]]),  # Meter 1
            torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),  # Meter 0
            torch.tensor([[3, 2, 1], [1, 3, 2], [1, 3, 2]]),  # Meter 1
        ]

        # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
        targets = [
            torch.tensor([0, 1, 2]),  # Meter 0
            torch.tensor([0, 1, 2]),  # Meter 1
            torch.tensor([0, 1, 2]),  # Meter 0
            torch.tensor([0, 1, 2]),  # Meter 1
        ]

        # In first two updates there are 3 correct top-2, 5 correct in top 2
        # The same occurs in the second two updates and is added to first
        expected_values = [
            {"top_1": 3 / 6.0, "top_2": 5 / 6.0},  # After one update to each meter
            {"top_1": 6 / 12.0, "top_2": 10 / 12.0},  # After two updates to each meter
        ]

        self.meter_distributed_test(meters, model_outputs, targets, expected_values)
