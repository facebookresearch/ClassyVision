#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Utility function for weight initialization"""

import torch.nn as nn


# from fvcore.nn.weight_init import c2_msra_fill


def init_weights(model, fc_init_std=0.01, zero_init_final_bn=True):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            pass
            # c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d):
            if (
                hasattr(m, "transform_final_bn")
                and m.transform_final_bn
                and zero_init_final_bn
            ):
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            m.weight.data.fill_(batchnorm_weight)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            m.bias.data.zero_()
