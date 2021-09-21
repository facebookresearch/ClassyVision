#!/usr/bin/env python3
# Portions Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 Ross Wightman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py

import math
from collections import abc
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.distributions.beta import Beta


def one_hot(x, num_classes, on_value=1.0, off_value=0.0, device="cuda"):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(
        1, x, on_value
    )


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0):
    if target.ndim == 1:
        off_value = smoothing / num_classes
        on_value = 1.0 - smoothing + off_value
        y1 = one_hot(
            target,
            num_classes,
            on_value=on_value,
            off_value=off_value,
            device=target.device,
        )
        y2 = one_hot(
            target.flip(0),
            num_classes,
            on_value=on_value,
            off_value=off_value,
            device=target.device,
        )
    else:
        # when 2D one-hot/multi-hot target tensor is already provided, skip label
        # smoothing
        assert target.ndim == 2, "target tensor shape must be 1D or 2D"
        y1 = target
        y2 = target.flip(0)

    return y1 * lam + y2 * (1.0 - lam)


def rand_bbox(img_shape, lam, margin=0.0, count=1):
    """Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = math.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = torch.randint(0 + margin_y, img_h - margin_y, (count,))
    cx = torch.randint(0 + margin_x, img_w - margin_x, (count,))
    yl = torch.clamp(cy - cut_h // 2, 0, img_h)
    yh = torch.clamp(cy + cut_h // 2, 0, img_h)
    xl = torch.clamp(cx - cut_w // 2, 0, img_w)
    xh = torch.clamp(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=1):
    """Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(
        int(img_h * minmax[0]), int(img_h * minmax[1]), size=count
    )
    cut_w = np.random.randint(
        int(img_w * minmax[0]), int(img_w * minmax[1]), size=count
    )
    # torch's randint does not accept a vector of max values
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return [torch.from_numpy(a) for a in [yl, yu, xl, xu]]


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=1):
    """Generate bbox and apply lambda correction."""
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = (1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])).item()

    return (yl, yu, xl, xu), lam


def _recursive_mixup(sample: Any, coeff: float):
    if isinstance(sample, (tuple, list)):
        mixed_sample = []
        for s in sample:
            mixed_sample.append(_recursive_mixup(s, coeff))

        return mixed_sample if isinstance(sample, list) else tuple(mixed_sample)
    elif isinstance(sample, abc.Mapping):
        mixed_sample = {}
        for key, val in sample.items():
            mixed_sample[key] = _recursive_mixup(val, coeff)

        return mixed_sample
    else:
        assert torch.is_tensor(sample), "sample is expected to be a pytorch tensor"
        # Assume training data is at least 3D tensor (i.e. 1D data). We only
        # mixup content data tensor (e.g. video clip, audio spectrogram), and skip
        # other tensors, such as frame_idx and timestamp in video clip samples.
        if sample.ndim >= 3:
            sample = coeff * sample + (1.0 - coeff) * sample.flip(0)

        return sample


class MixupTransform:
    """
    This implements the mixup data augmentation in the paper
    "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412)
    """

    def __init__(
        self,
        mixup_alpha: float,
        num_classes: Optional[int] = None,
        cutmix_alpha: float = 0,
        cutmix_minmax: Optional[Tuple[float]] = None,
        mix_prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = "batch",
        correct_lam: bool = True,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            mixup_alpha: the hyperparameter of Beta distribution used to sample mixup
            coefficient.
            num_classes: number of classes in the dataset.
            cutmix_alpha: cutmix alpha value, cutmix is active if > 0.
            cutmix_minmax cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
            mix_prob: probability of applying mixup or cutmix per batch or element
            switch_prob: probability of switching to cutmix instead of mixup when both are active
            mode: how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
            correct_lam: apply lambda correction when cutmix bbox clipped by image borders.
            label_smoothing: apply label smoothing to the mixed target tensor
        """
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        self.mix_prob = mix_prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.mode = mode
        self.correct_lam = correct_lam
        self.label_smoothing = label_smoothing

    def _params_per_elem(self, batch_size):
        lam = torch.ones(batch_size)
        use_cutmix = torch.zeros(batch_size, dtype=torch.bool)

        if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
            use_cutmix = torch.rand(batch_size) < self.switch_prob
            lam_mix = torch.where(
                use_cutmix,
                Beta(self.cutmix_alpha, self.cutmix_alpha).sample((batch_size,)),
                Beta(self.mixup_alpha, self.mixup_alpha).sample((batch_size,)),
            )
        elif self.mixup_alpha > 0.0:
            lam_mix = Beta(self.mixup_alpha, self.mixup_alpha).sample((batch_size,))
        elif self.cutmix_alpha > 0.0:
            use_cutmix = torch.ones(batch_size, dtype=torch.bool)
            lam_mix = Beta(self.cutmix_alpha, self.cutmix_alpha).sample((batch_size,))
        else:
            raise ValueError(
                "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            )

        lam = torch.where(torch.rand(batch_size) < self.mix_prob, lam_mix, lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.0
        use_cutmix = False
        if torch.rand(1) < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = torch.rand(1) < self.switch_prob
                lam_mix = (
                    Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
                    if use_cutmix
                    else Beta(self.mixup_alpha, self.mixup_alpha).sample()
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = Beta(self.mixup_alpha, self.mixup_alpha).sample()
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
            else:
                raise ValueError(
                    "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
                )
            lam = float(lam_mix)

        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam,
                    )
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)

        return lam_batch.to(x).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam,
                    )
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)

        lam_batch = torch.cat((lam_batch, lam_batch.flip(0)))
        return lam_batch.to(x).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape,
                lam,
                ratio_minmax=self.cutmix_minmax,
                correct_lam=self.correct_lam,
            )
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            sample: the batch data.
        """
        assert len(sample["target"]) % 2 == 0, "Batch size should be even"

        if torch.is_tensor(sample["input"]) and sample["input"].ndim == 4:
            # This is the simple case of image data batch (i.e. 4D tensor).
            # We support more advanved joint mixup and cutmix in this case.
            if self.mode == "elem":
                lam = self._mix_elem(sample["input"])
            elif self.mode == "pair":
                lam = self._mix_pair(sample["input"])
            else:
                lam = self._mix_batch(sample["input"])

            sample["target"] = mixup_target(
                sample["target"],
                self.num_classes,
                lam=lam,
                smoothing=self.label_smoothing,
            )
        else:
            # This is the complex case of video data batch (i.e. 5D tensor) or more complex
            # data batch. We only support mixup augmentation in batch mode.
            if sample["target"].ndim == 1:
                assert (
                    self.num_classes is not None
                ), "num_classes is expected for 1D target"

                off_value = self.label_smoothing / self.num_classes
                on_value = 1.0 - self.label_smoothing + off_value

                sample["target"] = one_hot(
                    sample["target"],
                    self.num_classes,
                    on_value=on_value,
                    off_value=off_value,
                    device=sample["target"].device,
                )
            else:
                assert (
                    sample["target"].ndim == 2
                ), "target tensor shape must be 1D or 2D"

            c = Beta(self.mixup_alpha, self.mixup_alpha).sample()

            sample["target"] = c * sample["target"] + (1.0 - c) * sample["target"].flip(
                0
            )
            sample["input"] = _recursive_mixup(sample["input"], c)

        return sample
