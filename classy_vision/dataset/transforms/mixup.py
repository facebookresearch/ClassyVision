#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import abc
from typing import Any, Dict, Optional

import torch
from classy_vision.generic.util import convert_to_one_hot
from torch.distributions.beta import Beta


def _recursive_mixup(sample: Any, permuted_indices: torch.Tensor, coeff: float):
    if isinstance(sample, (tuple, list)):
        mixed_sample = []
        for s in sample:
            mixed_sample.append(_recursive_mixup(s, permuted_indices, coeff))

        return mixed_sample if isinstance(sample, list) else tuple(mixed_sample)
    elif isinstance(sample, abc.Mapping):
        mixed_sample = {}
        for key, val in sample.items():
            mixed_sample[key] = _recursive_mixup(val, permuted_indices, coeff)

        return mixed_sample
    else:
        assert torch.is_tensor(sample), "sample is expected to be a pytorch tensor"
        # Assume training data is at least 3D tensor (i.e. 1D data). We only
        # mixup content data tensor (e.g. video clip, audio spectrogram), and skip
        # other tensors, such as frame_idx and timestamp in video clip samples.
        if sample.ndim >= 3:
            sample = coeff * sample + (1.0 - coeff) * sample[permuted_indices, :]

        return sample


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

        sample["target"] = (
            c * sample["target"] + (1.0 - c) * sample["target"][permuted_indices, :]
        )
        sample["input"] = _recursive_mixup(sample["input"], permuted_indices, c)

        return sample
