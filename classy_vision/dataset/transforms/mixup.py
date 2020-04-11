#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
from classy_vision.generic.util import convert_to_one_hot
from torch.distributions.beta import Beta


class MixupTransform:
    """
    This implements the mixup data augmentation in the paper
    "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412)
    """

    def __init__(self, alpha: float, num_classes: Optional[int] = None):
        """
        Args:
            alpha: the hyperparameter of Beta distribution used to sample mixup
            coefficient.
            num_classes: number of classes in the dataset.
        """
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            sample: the batch data.
        """
        if sample["target"].ndim == 1:
            assert self.num_classes is not None, "num_classes is expected for 1D target"
            sample["target"] = convert_to_one_hot(
                sample["target"].view(-1, 1), self.num_classes
            )
        else:
            assert sample["target"].ndim == 2, "target tensor shape must be 1D or 2D"

        c = Beta(self.alpha, self.alpha).sample().to(device=sample["target"].device)
        permuted_indices = torch.randperm(sample["target"].shape[0])
        for key in ["input", "target"]:
            sample[key] = c * sample[key] + (1.0 - c) * sample[key][permuted_indices, :]

        return sample
