#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""VGG models."""

# TODO(ilijar): Refactor wieght init to reduce duplication

import logging
import math

import torch
import torch.nn as nn
from classy_vision.generic.util import is_pos_int

from . import register_model
from .classy_model import ClassyModel


# Stage configurations (supports VGG11 VGG13, VGG16 and VGG19)
_STAGES = {
    11: [[64], [128], [256, 256], [512, 512], [512, 512]],
    13: [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    16: [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    19: [
        [64, 64],
        [128, 128],
        [256, 256, 256, 256],
        [512, 512, 512, 512],
        [512, 512, 512, 512],
    ],
}


@register_model("vgg")
class VGG(ClassyModel):
    """VGG model."""

    def __init__(
        self,
        num_classes,
        freeze_trunk,
        depth,
        num_stages,
        stride2_inds,
        max_pool_inds,
        ds_mult,
        ws_mult,
        bn_epsilon,
        bn_momentum,
        relu_inplace,
        small_input,
    ):
        super().__init__(num_classes, freeze_trunk)

        assert num_classes is None or is_pos_int(num_classes)
        assert depth in _STAGES.keys(), "VGG{} not supported".format(depth)
        assert is_pos_int(num_stages) and num_stages <= 5
        assert type(stride2_inds) == list and type(max_pool_inds) == list
        assert type(small_input) == bool and type(relu_inplace) == bool
        assert type(bn_epsilon) == float and type(bn_momentum) == float
        assert type(ds_mult) == float and type(ws_mult) == float

        logging.info(
            "Constructing: VGG-{}, stgs {}, s2c {}, mp {}, ds {}, ws {}".format(
                depth, num_stages, stride2_inds, max_pool_inds, ds_mult, ws_mult
            )
        )

        # Retrieve the original body structure
        stages = _STAGES[depth][:num_stages]

        # Adjust the widths and depths
        stages = [[int(stage[0] * ws_mult)] * len(stage) for stage in stages]
        stages = [[stage[0]] * int(len(stage) * ds_mult) for stage in stages]

        # Construct the body
        body_layers = []
        dim_in = 3

        for i, stage in enumerate(stages):
            # Construct the blocks for the stage
            for j, dim_out in enumerate(stage):
                # Determine the block stride
                stride = 2 if j == 0 and i in stride2_inds else 1
                # Basic block: Conv, BN, ReLU
                body_layers += [
                    nn.Conv2d(
                        dim_in,
                        dim_out,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(dim_out, eps=bn_epsilon, momentum=bn_momentum),
                    nn.ReLU(inplace=relu_inplace),
                ]
                dim_in = dim_out
            # Perform reduction by pooling
            if i in max_pool_inds:
                body_layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        self.body = nn.Sequential(*body_layers)

        # Construct the head
        self.head = nn.AdaptiveAvgPool2d((1, 1))

        # Construct the classifier layer
        self.num_planes = dim_in
        self.classifier = (
            None
            if num_classes is None
            else nn.Sequential(
                nn.Linear(self.num_planes, 4096),
                nn.ReLU(inplace=relu_inplace),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=relu_inplace),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        )

        self.small_input = small_input
        self.depth = depth

        self._init_weights()

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        del config["name"]
        assert all(
            e in config
            for e in [
                "depth",
                "num_stages",
                "stride2_inds",
                "max_pool_inds",
                "ds_mult",
                "ws_mult",
                "bn_epsilon",
                "bn_momentum",
                "relu_inplace",
            ]
        )

        config.setdefault("num_classes", None)
        config.setdefault("freeze_trunk", False)
        return cls(**config)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.body(x)
        out = torch.flatten(self.head(out))
        if self.classifier is not None:
            out = self.classifier(out)
        return out

    @property
    def input_shape(self):
        if self.small_input:
            return (3, 32, 32)
        else:
            return (3, 224, 224)

    @property
    def output_shape(self):
        return (1, self.num_classes)

    @property
    def model_depth(self):
        return self.depth

    def validate(self, dataset_output_shape):
        return self.input_shape == dataset_output_shape
