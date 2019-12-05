#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: Some batch-normalization operations are superfluous and can be removed.

# dependencies:
import math
from typing import Any, Dict

import torch
import torch.nn as nn
from classy_vision.generic.util import is_pos_int

from . import register_model
from .classy_model import ClassyModel


# global setting for in-place ReLU:
INPLACE = True


class _DenseLayer(nn.Sequential):
    """
        Single layer of a DenseNet.
    """

    def __init__(self, in_planes, growth_rate=32, expansion=4):

        # assertions:
        assert is_pos_int(in_planes)
        assert is_pos_int(growth_rate)
        assert is_pos_int(expansion)

        # add all layers to layer
        super(_DenseLayer, self).__init__()
        intermediate = expansion * growth_rate
        self.add_module("norm-1", nn.BatchNorm2d(in_planes))
        self.add_module("relu-1", nn.ReLU(inplace=INPLACE))
        self.add_module(
            "conv-1",
            nn.Conv2d(in_planes, intermediate, kernel_size=1, stride=1, bias=False),
        )
        self.add_module("norm-2", nn.BatchNorm2d(intermediate))
        self.add_module("relu-2", nn.ReLU(inplace=INPLACE))
        self.add_module(
            "conv-2",
            nn.Conv2d(
                intermediate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """
        Block of densely connected layers at same resolution.
    """

    def __init__(self, num_layers, in_planes, growth_rate=32, expansion=4):

        # assertions:
        assert is_pos_int(in_planes)
        assert is_pos_int(growth_rate)
        assert is_pos_int(expansion)

        # create block of dense layers at same resolution:
        super(_DenseBlock, self).__init__()
        for idx in range(num_layers):
            layer = _DenseLayer(
                in_planes + idx * growth_rate,
                growth_rate=growth_rate,
                expansion=expansion,
            )
            self.add_module("denselayer-%d" % (idx + 1), layer)


class _Transition(nn.Sequential):
    """
        Transition layer to reduce spatial resolution.
    """

    def __init__(self, in_planes, out_planes, reduction=2):

        # assertions:
        assert is_pos_int(in_planes)
        assert is_pos_int(out_planes)
        assert is_pos_int(reduction)

        # create layers for pooling:
        super(_Transition, self).__init__()
        self.add_module("pool-norm", nn.BatchNorm2d(in_planes))
        self.add_module("pool-relu", nn.ReLU(inplace=INPLACE))
        self.add_module(
            "pool-conv",
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
        )
        self.add_module(
            "pool-pool", nn.AvgPool2d(kernel_size=reduction, stride=reduction)
        )


@register_model("densenet")
class DenseNet(ClassyModel):
    def __init__(
        self,
        num_blocks,
        num_classes,
        init_planes,
        growth_rate,
        expansion,
        small_input,
        final_bn_relu,
    ):
        """
            Implementation of a standard densely connected network (DenseNet).

            Set `small_input` to `True` for 32x32 sized image inputs.

            Set `final_bn_relu` to `False` to exclude the final batchnorm and ReLU
            layers. These settings are useful when
            training Siamese networks.
        """
        super().__init__()

        # assertions:
        assert type(num_blocks) == list
        assert all(is_pos_int(b) for b in num_blocks)
        assert num_classes is None or is_pos_int(num_classes)
        assert is_pos_int(init_planes)
        assert is_pos_int(growth_rate)
        assert is_pos_int(expansion)
        assert type(small_input) == bool

        # initial convolutional block:
        self._num_classes = num_classes
        self.num_blocks = num_blocks
        self.small_input = small_input
        if self.small_input:
            self.initial_block = nn.Sequential(
                nn.Conv2d(
                    3, init_planes, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.BatchNorm2d(init_planes),
                nn.ReLU(inplace=INPLACE),
            )
        else:
            self.initial_block = nn.Sequential(
                nn.Conv2d(
                    3, init_planes, kernel_size=7, stride=2, padding=3, bias=False
                ),
                nn.BatchNorm2d(init_planes),
                nn.ReLU(inplace=INPLACE),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        # loop over spatial resolutions:
        num_planes = init_planes
        self.features = nn.Sequential()
        for idx, num_layers in enumerate(num_blocks):

            # add dense block:
            block = _DenseBlock(
                num_layers, num_planes, growth_rate=growth_rate, expansion=expansion
            )
            self.features.add_module("denseblock-%d" % (idx + 1), block)
            num_planes = num_planes + num_layers * growth_rate

            # add transition layer:
            if idx != len(num_blocks) - 1:
                trans = _Transition(num_planes, num_planes // 2)
                self.features.add_module("transition-%d" % (idx + 1), trans)
                num_planes = num_planes // 2

        # final batch normalization:
        if final_bn_relu:
            self.features.add_module("norm-final", nn.BatchNorm2d(num_planes))
            self.features.add_module("relu-final", nn.ReLU(inplace=INPLACE))

        # final classifier:
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None if num_classes is None else nn.Linear(num_planes, num_classes)
        self.num_planes = num_planes

        # initialize weights of convolutional and batchnorm layers:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DenseNet":
        """Instantiates a DenseNet from a configuration.

        Args:
            config: A configuration for a DenseNet.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A DenseNet instance.
        """
        assert "num_blocks" in config
        config = {
            "num_blocks": config["num_blocks"],
            "num_classes": config.get("num_classes"),
            "init_planes": config.get("init_planes", 64),
            "growth_rate": config.get("growth_rate", 32),
            "expansion": config.get("expansion", 4),
            "small_input": config.get("small_input", False),
            "final_bn_relu": config.get("final_bn_relu", True),
        }
        return cls(**config)

    # forward pass in DenseNet:
    def forward(self, x):

        # initial convolutional block:
        out = self.initial_block(x)

        # evaluate all dense blocks:
        out = self.features(out)

        # perform average pooling:
        out = self.avgpool(out)

        # final classifier:
        out = out.view(out.size(0), -1)
        if self.fc is not None:
            out = self.fc(out)
        return out

    def get_optimizer_params(self):
        # use weight decay on BatchNorm for DenseNets
        return super().get_optimizer_params(bn_weight_decay=True)

    @property
    def input_shape(self):
        if self.small_input:
            return (3, 32, 32)
        else:
            return (3, 224, 224)

    @property
    def output_shape(self):
        return (1, self._num_classes)

    @property
    def model_depth(self):
        return sum(self.num_blocks)
