#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch.nn as nn
from classy_vision.generic.util import is_pos_int
from classy_vision.heads import ClassyHead, register_head


class FullyConvolutionalLinear(nn.Module):
    """
    FC layer in ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1.
    """

    def __init__(self, dim_in, num_classes, act_func="softmax"):
        """
        Args:
            dim_in (int): No. channels of input tensor
            num_classes (int): No. channels of output tensor
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(FullyConvolutionalLinear, self).__init__()
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        x = self.projection(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.view(x.shape[0], -1)
        return x


@register_head("fully_convolutional_linear")
class FullyConvolutionalLinearHead(ClassyHead):
    def __init__(
        self, unique_id, num_classes, in_plane, pool_size, activation_func, use_dropout
    ):
        super().__init__(unique_id, num_classes)
        self.final_avgpool = nn.AvgPool3d(pool_size, stride=1)
        if use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        # we separate average pooling from the fully-convolutional linear projection
        # because for multi-path models such as SlowFast model, the input can be
        # more than 1 tesnor. In such case, we can define a new head to combine multiple
        # tensors via concat or addition, do average pooling, but still reuse
        # FullyConvolutionalLinear inside of it.
        self.head_fcl = FullyConvolutionalLinear(
            in_plane, num_classes, act_func=activation_func
        )

    @classmethod
    def from_config(cls, config):
        required_args = ["pool_size", "in_plane", "num_classes"]
        for arg in required_args:
            assert arg in config, "argument %s is required" % arg

        config.update({"activation_func": config.get("activation_func", "softmax")})
        config.update({"use_dropout": config.get("use_dropout", False)})

        assert (
            isinstance(config["pool_size"], Sequence) and len(config["pool_size"]) == 3
        )
        for pool_size_dim in config["pool_size"]:
            assert is_pos_int(pool_size_dim)
        assert is_pos_int(config["in_plane"])
        assert is_pos_int(config["num_classes"])

        num_classes = config.get("num_classes", None)
        in_plane = config["in_plane"]
        return cls(
            config["unique_id"],
            num_classes,
            in_plane,
            config["pool_size"],
            config["activation_func"],
            config["use_dropout"],
        )

    def forward(self, x):
        out = self.final_avgpool(x)
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out = self.head_fcl(out)
        return out
