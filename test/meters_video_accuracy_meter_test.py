#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from test.generic.meter_test_utils import ClassificationMeterTest

import torch
from classy_vision import meters
from classy_vision.meters import VideoAccuracyMeter


class TestVideoAccuracyMeter(ClassificationMeterTest):
    def test_accuracy_meter_registry(self):
        accuracy_meter = meters.build_meter(
            {
                "name": "video_accuracy",
                "topk": [1, 2],
                "clips_per_video_train": 1,
                "clips_per_video_test": 2,
            }
        )
        self.assertTrue(isinstance(accuracy_meter, VideoAccuracyMeter))

    def test_single_meter_update_and_reset(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update.
        """
        meter = VideoAccuracyMeter(
            topk=[1, 2], clips_per_video_train=1, clips_per_video_test=2
        )
        # Batchsize = 3, num classes = 3, clips_per_video is 2,
        # score is a value in {1, 2, 3}
        model_output = torch.tensor(
            [[3, 2, 1], [3, 1, 2], [1, 2, 2], [1, 2, 3], [2, 2, 2], [1, 3, 2]],
            dtype=torch.float,
        )
        # Class 0 is the correct class for video 1, class 2 for video 2, and
        # class 1 for video
        target = torch.tensor([0, 0, 1, 1, 2, 2])

        # Only the first sample has top class correct, first and third
        # sample have correct class in top 2
        expected_value = {"top_1": 1 / 3.0, "top_2": 3 / 3.0}

        self.meter_update_and_reset_test(
            meter, model_output, target, expected_value, is_train=False
        )

    def test_double_meter_update_and_reset(self):
        meter = VideoAccuracyMeter(
            topk=[1, 2], clips_per_video_train=1, clips_per_video_test=2
        )
        # Batchsize = 3, num classes = 3, clips_per_video is 2,
        # score is a value in {1, 2, 3}.
        # Data of two batch is provided
        model_outputs = [
            torch.tensor(
                [[3, 2, 1], [3, 1, 2], [1, 2, 2], [1, 2, 3], [2, 2, 2], [1, 3, 2]],
                dtype=torch.float,
            ),
            torch.tensor(
                [[3, 2, 1], [3, 1, 2], [1, 2, 2], [1, 2, 3], [2, 2, 2], [1, 3, 2]],
                dtype=torch.float,
            ),
        ]
        # Class 0 is the correct class for video 1, class 2 for video 2, and
        # class 1 for video, in both batches
        targets = [torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 0, 1, 1, 2, 2])]

        # First batch has top-1 accuracy of 1/3.0, top-2 accuracy of 2/3.0
        # Second batch has top-1 accuracy of 2/3.0, top-2 accuracy of 3/3.0
        expected_value = {"top_1": 2 / 6.0, "top_2": 6 / 6.0}

        self.meter_update_and_reset_test(
            meter, model_outputs, targets, expected_value, is_train=False
        )

    def test_meter_invalid_model_output(self):
        meter = VideoAccuracyMeter(
            topk=[1, 2], clips_per_video_train=1, clips_per_video_test=2
        )
        # This model output has 3 dimensions instead of expected 2
        model_output = torch.tensor(
            [[[3, 2, 1], [1, 2, 3]], [[-1, -3, -4], [-10, -90, -100]]],
            dtype=torch.float,
        )
        target = torch.tensor([0, 1, 2])

        self.meter_invalid_meter_input_test(meter, model_output, target)

    def test_meter_invalid_target(self):
        meter = VideoAccuracyMeter(
            topk=[1, 2], clips_per_video_train=1, clips_per_video_test=2
        )
        model_output = torch.tensor(
            [[3, 2, 1], [3, 1, 2], [1, 2, 2], [1, 2, 3], [2, 2, 2], [1, 3, 2]],
            dtype=torch.float,
        )
        # Target has 2 dimensions instead of expected 1
        target = torch.tensor([[0, 1, 2], [0, 1, 2]])

        self.meter_invalid_meter_input_test(meter, model_output, target)
        # Target of clips from the same video is not consistent
        target = torch.tensor([0, 2, 1, 1, 2, 2])

        self.meter_invalid_update_test(meter, model_output, target, is_train=False)

    def test_meter_invalid_topk(self):
        meter = VideoAccuracyMeter(
            topk=[1, 5], clips_per_video_train=1, clips_per_video_test=2
        )
        model_output = torch.tensor(
            [[3, 2, 1], [3, 1, 2], [1, 2, 2], [1, 2, 3], [2, 2, 2], [1, 3, 2]],
            dtype=torch.float,
        )
        target = torch.tensor([0, 1, 2])
        self.meter_invalid_meter_input_test(meter, model_output, target)

    def test_meter_get_set_classy_state_test(self):
        # In this test we update meter0 with model_output0 & target0
        # and we update meter1 with model_output1 & target1 then
        # transfer the state from meter1 to meter0 and validate they
        # give same expected value.
        # Expected value is the expected value of meter1
        meters = [
            VideoAccuracyMeter(
                topk=[1, 2], clips_per_video_train=1, clips_per_video_test=2
            ),
            VideoAccuracyMeter(
                topk=[1, 2], clips_per_video_train=1, clips_per_video_test=2
            ),
        ]
        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_outputs = [
            torch.tensor(
                [[1, 2, 3], [1, 1, 3], [2, 2, 1], [3, 2, 1], [2, 2, 2], [2, 3, 1]],
                dtype=torch.float,
            ),
            torch.tensor(
                [[3, 2, 1], [3, 1, 2], [1, 2, 2], [1, 2, 3], [2, 2, 2], [1, 3, 2]],
                dtype=torch.float,
            ),
        ]
        # Class 2 is the correct class for sample 1, class 0 for sample 2, etc
        targets = [torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 0, 1, 1, 2, 2])]
        # Value for second update
        expected_value = {"top_1": 1 / 3.0, "top_2": 3 / 3.0}

        self.meter_get_set_classy_state_test(
            meters, model_outputs, targets, expected_value, is_train=False
        )

    def test_meter_distributed(self):
        # Meter0 will execute on one process, Meter1 on the other
        meters = [
            VideoAccuracyMeter(
                topk=[1, 2], clips_per_video_train=1, clips_per_video_test=2
            ),
            VideoAccuracyMeter(
                topk=[1, 2], clips_per_video_train=1, clips_per_video_test=2
            ),
        ]

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_outputs = [
            torch.tensor(
                [[1, 2, 3], [1, 1, 3], [2, 2, 1], [3, 2, 1], [2, 2, 2], [2, 3, 1]],
                dtype=torch.float,
            ),  # Meter 0
            torch.tensor(
                [[3, 2, 1], [3, 1, 2], [1, 2, 2], [1, 2, 3], [2, 2, 2], [1, 3, 2]],
                dtype=torch.float,
            ),  # Meter 1
            torch.tensor(
                [[1, 2, 3], [1, 1, 3], [2, 2, 1], [3, 2, 1], [2, 2, 2], [2, 3, 1]],
                dtype=torch.float,
            ),  # Meter 0
            torch.tensor(
                [[3, 2, 1], [3, 1, 2], [1, 2, 2], [1, 2, 3], [2, 2, 2], [1, 3, 2]],
                dtype=torch.float,
            ),  # Meter 1
        ]

        # For meter 0, class 2 is the correct class for sample 1, class 0 for sample 2,
        # etc
        targets = [
            torch.tensor([0, 0, 1, 1, 2, 2]),  # Meter 0
            torch.tensor([0, 0, 1, 1, 2, 2]),  # Meter 1
            torch.tensor([0, 0, 1, 1, 2, 2]),  # Meter 0
            torch.tensor([0, 0, 1, 1, 2, 2]),  # Meter 1
        ]

        # In first two updates there are 3 correct top-2, 5 correct in top 2
        # The same occurs in the second two updates and is added to first
        expected_values = [
            {"top_1": 1 / 6.0, "top_2": 4 / 6.0},  # After one update to each meter
            {"top_1": 2 / 12.0, "top_2": 8 / 12.0},  # After two updates to each meter
        ]

        self.meter_distributed_test(
            meters, model_outputs, targets, expected_values, is_train=False
        )
