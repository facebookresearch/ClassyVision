#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch.nn as nn


class SqueezeAndExcitationLayer(nn.Module):
    """Squeeze and excitation layer, as per https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(
        self,
        in_planes,
        reduction_ratio: Optional[int] = 16,
        reduced_planes: Optional[int] = None,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Either reduction_ratio is defined, or out_planes is defined,
        # neither both nor none of them
        assert bool(reduction_ratio) != bool(reduced_planes)

        reduced_planes = (
            in_planes // reduction_ratio if reduced_planes is None else reduced_planes
        )
        self.excitation = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_squeezed = self.avgpool(x)
        x_excited = self.excitation(x_squeezed)
        x_scaled = x * x_excited
        return x_scaled
