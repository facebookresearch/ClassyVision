#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import register_transform
from .classy_transform import ClassyTransform


_IMAGENET_EIGEN_VAL = [0.2175, 0.0188, 0.0045]
_IMAGENET_EIGEN_VEC = [
    [-144.7125, 183.396, 102.2295],
    [-148.104, -1.1475, -207.57],
    [-148.818, -177.174, 107.1765],
]

_DEFAULT_COLOR_LIGHTING_STD = 0.1


@register_transform("lighting")
class LightingTransform(ClassyTransform):
    """
    Lighting noise(AlexNet - style PCA - based noise).
    This trick was originally used in `AlexNet paper
    <https://papers.nips.cc/paper/4824-imagenet-classification
    -with-deep-convolutional-neural-networks.pdf>`_

    The eigen values and eigen vectors, are taken from caffe2 `ImageInputOp.h
    <https://github.com/pytorch/pytorch/blob/master/caffe2/image/
    image_input_op.h#L265>`_.
    """

    def __init__(
        self,
        alphastd=_DEFAULT_COLOR_LIGHTING_STD,
        eigval=_IMAGENET_EIGEN_VAL,
        eigvec=_IMAGENET_EIGEN_VEC,
    ):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval)
        # Divide by 255 as the Lighting operation is expected to be applied
        # on `img` pixels ranging between [0.0, 1.0]
        self.eigvec = torch.tensor(eigvec) / 255.0

    def __call__(self, img):
        """
        img: (C x H x W) Tensor with values in range [0.0, 1.0]
        """
        assert (
            img.min() >= 0.0 and img.max() <= 1.0
        ), "Image should be normalized by 255 and be in range [0.0, 1.0]"
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))
