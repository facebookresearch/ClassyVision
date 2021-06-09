#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from typing import Any, Dict

import numpy as np
import torch.nn as nn
from classy_vision.models import register_model

from .anynet import (
    AnyNet,
    AnyNetParams,
    StemType,
    BlockType,
    ActivationType,
    RELU_IN_PLACE,
)


def _quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def _adjust_widths_groups_compatibilty(stage_widths, bottleneck_ratios, group_widths):
    """Adjusts the compatibility of widths and groups,
    depending on the bottleneck ratio."""
    # Compute all widths for the current settings
    widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
    groud_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

    # Compute the adjusted widths so that stage and group widths fit
    ws_bot = [_quantize_float(w_bot, g) for w_bot, g in zip(widths, groud_widths_min)]
    stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
    return stage_widths, groud_widths_min


class RegNetParams(AnyNetParams):
    def __init__(
        self,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        stem_type: StemType = StemType.SIMPLE_STEM_IN,
        stem_width: int = 32,
        block_type: BlockType = BlockType.RES_BOTTLENECK_BLOCK,
        activation: ActivationType = ActivationType.RELU,
        use_se: bool = True,
        se_ratio: float = 0.25,
        bn_epsilon: float = 1e-05,
        bn_momentum: bool = 0.1,
    ):
        assert (
            w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % 8 == 0
        ), "Invalid RegNet settings"
        self.depth = depth
        self.w_0 = w_0
        self.w_a = w_a
        self.w_m = w_m
        self.group_width = group_width
        self.bottleneck_multiplier = bottleneck_multiplier
        self.stem_type = stem_type
        self.block_type = block_type
        self.activation = activation
        self.stem_width = stem_width
        self.use_se = use_se
        self.se_ratio = se_ratio if use_se else None
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.relu_in_place = RELU_IN_PLACE

    def get_expanded_params(self):
        """Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage
        """

        QUANT = 8
        STRIDE = 2

        # Compute the block widths. Each stage has one unique block width
        widths_cont = np.arange(self.depth) * self.w_a + self.w_0
        block_capacity = np.round(np.log(widths_cont / self.w_0) / np.log(self.w_m))
        block_widths = (
            np.round(np.divide(self.w_0 * np.power(self.w_m, block_capacity), QUANT))
            * QUANT
        )
        num_stages = len(np.unique(block_widths))
        block_widths = block_widths.astype(int).tolist()

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = np.diff([d for d, t in enumerate(splits) if t]).tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [self.bottleneck_multiplier] * num_stages
        group_widths = [self.group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = _adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return zip(
            stage_widths, strides, stage_depths, group_widths, bottleneck_multipliers
        )


@register_model("regnet")
class RegNet(AnyNet):
    """Implementation of RegNet, a particular form of AnyNets.

    See https://arxiv.org/abs/2003.13678 for introduction to RegNets, and details about
    RegNetX and RegNetY models.

    See https://arxiv.org/abs/2103.06877 for details about RegNetZ models.
    """

    def __init__(self, params: RegNetParams):
        super().__init__(params)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RegNet":
        """Instantiates a RegNet from a configuration.

        Args:
            config: A configuration for a RegNet.
                See `RegNetParams` for parameters expected in the config.

        Returns:
            A RegNet instance.
        """

        params = RegNetParams(
            depth=config["depth"],
            w_0=config["w_0"],
            w_a=config["w_a"],
            w_m=config["w_m"],
            group_width=config["group_width"],
            bottleneck_multiplier=config.get("bottleneck_multiplier", 1.0),
            stem_type=StemType[config.get("stem_type", "simple_stem_in").upper()],
            stem_width=config.get("stem_width", 32),
            block_type=BlockType[
                config.get("block_type", "res_bottleneck_block").upper()
            ],
            activation=ActivationType[config.get("activation", "relu").upper()],
            use_se=config.get("use_se", True),
            se_ratio=config.get("se_ratio", 0.25),
            bn_epsilon=config.get("bn_epsilon", 1e-05),
            bn_momentum=config.get("bn_momentum", 0.1),
        )

        return cls(params)

    def forward(self, x, *args, **kwargs):
        x = self.stem(x)
        x = self.trunk_output(x)

        return x

    def init_weights(self):
        # Performs ResNet-style weight initialization
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


# Register some "classic" RegNets
class _RegNet(RegNet):
    def __init__(self, params: RegNetParams):
        super().__init__(params)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RegNet":
        config = copy.deepcopy(config)
        config.pop("name")
        if "heads" in config:
            config.pop("heads")
        return cls(**config)


@register_model("regnet_y_400mf")
class RegNetY400mf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 440 feature maps
        super().__init__(
            RegNetParams(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, **kwargs)
        )


@register_model("regnet_y_800mf")
class RegNetY800mf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 768 feature maps
        super().__init__(
            RegNetParams(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, **kwargs)
        )


@register_model("regnet_y_1.6gf")
class RegNetY1_6gf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 888 feature maps
        super().__init__(
            RegNetParams(
                depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, **kwargs
            )
        )


@register_model("regnet_y_3.2gf")
class RegNetY3_2gf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 1512 feature maps
        super().__init__(
            RegNetParams(
                depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, **kwargs
            )
        )


@register_model("regnet_y_8gf")
class RegNetY8gf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 2016 feature maps
        super().__init__(
            RegNetParams(
                depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, **kwargs
            )
        )


@register_model("regnet_y_16gf")
class RegNetY16gf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 3024 feature maps
        super().__init__(
            RegNetParams(
                depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, **kwargs
            )
        )


@register_model("regnet_y_32gf")
class RegNetY32gf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 3712 feature maps
        super().__init__(
            RegNetParams(
                depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, **kwargs
            )
        )


@register_model("regnet_x_400mf")
class RegNetX400mf(_RegNet):
    def __init__(self, **kwargs):
        super().__init__(
            RegNetParams(
                depth=22,
                w_0=24,
                w_a=24.48,
                w_m=2.54,
                group_width=16,
                use_se=False,
                **kwargs,
            )
        )


@register_model("regnet_x_800mf")
class RegNetX800mf(_RegNet):
    def __init__(self, **kwargs):
        super().__init__(
            RegNetParams(
                depth=16,
                w_0=56,
                w_a=35.73,
                w_m=2.28,
                group_width=16,
                use_se=False,
                **kwargs,
            )
        )


@register_model("regnet_x_1.6gf")
class RegNetX1_6gf(_RegNet):
    def __init__(self, **kwargs):
        super().__init__(
            RegNetParams(
                depth=18,
                w_0=80,
                w_a=34.01,
                w_m=2.25,
                group_width=24,
                use_se=False,
                **kwargs,
            )
        )


@register_model("regnet_x_3.2gf")
class RegNetX3_2gf(_RegNet):
    def __init__(self, **kwargs):
        super().__init__(
            RegNetParams(
                depth=25,
                w_0=88,
                w_a=26.31,
                w_m=2.25,
                group_width=48,
                use_se=False,
                **kwargs,
            )
        )


@register_model("regnet_x_8gf")
class RegNetX8gf(_RegNet):
    def __init__(self, **kwargs):
        super().__init__(
            RegNetParams(
                depth=23,
                w_0=80,
                w_a=49.56,
                w_m=2.88,
                group_width=120,
                use_se=False,
                **kwargs,
            )
        )


@register_model("regnet_x_16gf")
class RegNetX16gf(_RegNet):
    def __init__(self, **kwargs):
        super().__init__(
            RegNetParams(
                depth=22,
                w_0=216,
                w_a=55.59,
                w_m=2.1,
                group_width=128,
                use_se=False,
                **kwargs,
            )
        )


@register_model("regnet_x_32gf")
class RegNetX32gf(_RegNet):
    def __init__(self, **kwargs):
        super().__init__(
            RegNetParams(
                depth=23,
                w_0=320,
                w_a=69.86,
                w_m=2.0,
                group_width=168,
                use_se=False,
                **kwargs,
            )
        )


# note that RegNetZ models are trained with a convolutional head, i.e. the
# fully_connected ClassyHead with conv_planes > 0.
@register_model("regnet_z_500mf")
class RegNetZ500mf(_RegNet):
    def __init__(self, **kwargs):
        super().__init__(
            RegNetParams(
                depth=21,
                w_0=16,
                w_a=10.7,
                w_m=2.51,
                group_width=4,
                bottleneck_multiplier=4.0,
                block_type=BlockType.RES_BOTTLENECK_LINEAR_BLOCK,
                activation=ActivationType.SILU,
                **kwargs,
            )
        )


# this is supposed to be trained with a resolution of 256x256
@register_model("regnet_z_4gf")
class RegNetZ4gf(_RegNet):
    def __init__(self, **kwargs):
        super().__init__(
            RegNetParams(
                depth=28,
                w_0=48,
                w_a=14.5,
                w_m=2.226,
                group_width=8,
                bottleneck_multiplier=4.0,
                block_type=BlockType.RES_BOTTLENECK_LINEAR_BLOCK,
                activation=ActivationType.SILU,
                **kwargs,
            )
        )


# -----------------------------------------------------------------------------------
# The following models were not part of the original publication,
# (https://arxiv.org/abs/2003.13678v1), but are larger versions of the
# published models, obtained in the same manner.


@register_model("regnet_y_64gf")
class RegNetY64gf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 2976 feature maps
        super().__init__(
            RegNetParams(
                depth=20, w_0=352, w_a=147.48, w_m=2.4, group_width=328, **kwargs
            )
        )


@register_model("regnet_y_128gf")
class RegNetY128gf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 7392 feature maps
        super().__init__(
            RegNetParams(
                depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, **kwargs
            )
        )


@register_model("regnet_y_256gf")
class RegNetY256gf(_RegNet):
    def __init__(self, **kwargs):
        # Output size: 5088 feature maps
        super().__init__(
            RegNetParams(
                depth=27, w_0=640, w_a=124.47, w_m=2.04, group_width=848, **kwargs
            )
        )
