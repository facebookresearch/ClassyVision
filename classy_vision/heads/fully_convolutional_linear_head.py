#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import torch.nn as nn
from classy_vision.generic.util import is_pos_int
from classy_vision.heads import ClassyHead, register_head


class FullyConvolutionalLinear(nn.Module):
    def __init__(self, dim_in, num_classes, act_func="softmax"):
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
        x = x.flatten(start_dim=1)
        return x


@register_head("fully_convolutional_linear")
class FullyConvolutionalLinearHead(ClassyHead):
    """
    This head defines a 3d average pooling layer (:class:`torch.nn.AvgPool3d` or
    :class:`torch.nn.AdaptiveAvgPool3d` if pool_size is None) followed by a fully
    convolutional linear layer. This layer performs a fully-connected projection
    during training, when the input size is 1x1x1.
    It performs a convolutional projection during testing when the input size
    is larger than 1x1x1.
    """

    def __init__(
        self,
        unique_id: str,
        num_classes: int,
        in_plane: int,
        pool_size: Optional[List[int]],
        activation_func: str,
        use_dropout: Optional[bool] = None,
    ):
        """
        Constructor for FullyConvolutionalLinearHead.

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.
            num_classes: Number of classes for the head.
            in_plane: Input size for the fully connected layer.
            pool_size: Optional kernel size for the 3d pooling layer. If None, use
                :class:`torch.nn.AdaptiveAvgPool3d` with output size (1, 1, 1).
            activation_func: activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            use_dropout: Whether to apply dropout after the pooling layer.
        """
        super().__init__(unique_id, num_classes)
        if pool_size is not None:
            self.final_avgpool = nn.AvgPool3d(pool_size, stride=1)
        else:
            self.final_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
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
    def from_config(cls, config: Dict[str, Any]) -> "FullyConvolutionalLinearHead":
        """Instantiates a FullyConvolutionalLinearHead from a configuration.

        Args:
            config: A configuration for a FullyConvolutionalLinearHead.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A FullyConvolutionalLinearHead instance.
        """
        required_args = ["in_plane", "num_classes"]
        for arg in required_args:
            assert arg in config, "argument %s is required" % arg

        config.update({"activation_func": config.get("activation_func", "softmax")})
        config.update({"use_dropout": config.get("use_dropout", False)})

        pool_size = config.get("pool_size", None)
        if pool_size is not None:
            assert isinstance(pool_size, Sequence) and len(pool_size) == 3
            for pool_size_dim in pool_size:
                assert is_pos_int(pool_size_dim)

        assert is_pos_int(config["in_plane"])
        assert is_pos_int(config["num_classes"])

        num_classes = config.get("num_classes", None)
        in_plane = config["in_plane"]
        return cls(
            config["unique_id"],
            num_classes,
            in_plane,
            pool_size,
            config["activation_func"],
            config["use_dropout"],
        )

    def forward(self, x):
        out = self.final_avgpool(x)
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out = self.head_fcl(out)
        return out
