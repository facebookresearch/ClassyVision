#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch.nn as nn


def r2plus1_unit(
    dim_in,
    dim_out,
    temporal_stride,
    spatial_stride,
    groups,
    inplace_relu,
    bn_eps,
    bn_mmt,
    dim_mid=None,
):
    """
    Implementation of `R(2+1)D unit <https://arxiv.org/abs/1711.11248>`_.
    Decompose one 3D conv into one 2D spatial conv and one 1D temporal conv.
    Choose the middle dimensionality so that the total No. of parameters
    in 2D spatial conv and 1D temporal conv is unchanged.

    Args:
        dim_in (int): the channel dimensions of the input.
        dim_out (int): the channel dimension of the output.
        temporal_stride (int): the temporal stride of the bottleneck.
        spatial_stride (int): the spatial_stride of the bottleneck.
        groups (int): number of groups for the convolution.
        inplace_relu (bool): calculate the relu on the original input
            without allocating new memory.
        bn_eps (float): epsilon for batch norm.
        bn_mmt (float): momentum for batch norm. Noted that BN momentum in
            PyTorch = 1 - BN momentum in Caffe2.
        dim_mid (Optional[int]): If not None, use the provided channel dimension
            for the output of the 2D spatial conv. If None, compute the output
            channel dimension of the 2D spatial conv so that the total No. of
            model parameters remains unchanged.
    """
    if dim_mid is None:
        dim_mid = int(dim_out * dim_in * 3 * 3 * 3 / (dim_in * 3 * 3 + dim_out * 3))
        logging.info(
            "dim_in: %d, dim_out: %d. Set dim_mid to %d" % (dim_in, dim_out, dim_mid)
        )
    # 1x3x3 group conv, BN, ReLU
    conv_middle = nn.Conv3d(
        dim_in,
        dim_mid,
        [1, 3, 3],  # kernel
        stride=[1, spatial_stride, spatial_stride],
        padding=[0, 1, 1],
        groups=groups,
        bias=False,
    )
    conv_middle_bn = nn.BatchNorm3d(dim_mid, eps=bn_eps, momentum=bn_mmt)
    conv_middle_relu = nn.ReLU(inplace=inplace_relu)
    # 3x1x1 group conv
    conv = nn.Conv3d(
        dim_mid,
        dim_out,
        [3, 1, 1],  # kernel
        stride=[temporal_stride, 1, 1],
        padding=[1, 0, 0],
        groups=groups,
        bias=False,
    )
    return nn.Sequential(conv_middle, conv_middle_bn, conv_middle_relu, conv)
