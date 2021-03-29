#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from collections import OrderedDict
from enum import Enum, auto
from typing import Any, Dict, Optional

import numpy as np
import torch.nn as nn
from classy_vision.generic.util import get_torch_version
from classy_vision.models import ClassyModel, register_model
from classy_vision.models.squeeze_and_excitation_layer import SqueezeAndExcitationLayer


RELU_IN_PLACE = True


# The different possible blocks
class BlockType(Enum):
    VANILLA_BLOCK = auto()
    RES_BASIC_BLOCK = auto()
    RES_BOTTLENECK_BLOCK = auto()
    RES_BOTTLENECK_LINEAR_BLOCK = auto()


# The different possible Stems
class StemType(Enum):
    RES_STEM_CIFAR = auto()
    RES_STEM_IN = auto()
    SIMPLE_STEM_IN = auto()


# The different possible activations
class ActivationType(Enum):
    RELU = auto()
    SILU = auto()


class BasicTransform(nn.Sequential):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ):
        super().__init__()

        self.a = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
            nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False),
        )

        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 2


class ResStemCifar(nn.Sequential):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )
        self.depth = 2


class ResStemIN(nn.Sequential):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(width_in, width_out, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.depth = 3


class SimpleStemIN(nn.Sequential):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )
        self.depth = 2


class VanillaBlock(nn.Sequential):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.a = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        self.b = nn.Sequential(
            nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        self.depth = 2


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.proj_block = (width_in != width_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(
                width_in, width_out, 1, stride=stride, padding=0, bias=False
            )
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BasicTransform(
            width_in, width_out, stride, bn_epsilon, bn_momentum, activation
        )
        self.activation = activation

        # The projection and transform happen in parallel,
        # and ReLU is not counted with respect to depth
        self.depth = self.f.depth

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)

        return self.activation(x)


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ):
        super().__init__()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        self.a = nn.Sequential(
            nn.Conv2d(width_in, w_b, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        self.b = nn.Sequential(
            nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False),
            nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            self.se = SqueezeAndExcitationLayer(
                in_planes=w_b,
                reduction_ratio=None,
                reduced_planes=width_se_out,
                activation=activation,
            )

        self.c = nn.Conv2d(w_b, width_out, 1, stride=1, padding=0, bias=False)
        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 3 if not se_ratio else 4


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ):
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj_block = (width_in != width_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(
                width_in, width_out, 1, stride=stride, padding=0, bias=False
            )
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            bn_epsilon,
            bn_momentum,
            activation,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation

        # The projection and transform happen in parallel,
        # and activation is not counted with respect to depth
        self.depth = self.f.depth

    def forward(self, x, *args):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class ResBottleneckLinearBlock(nn.Module):
    """Residual linear bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int = 1,
        bottleneck_multiplier: float = 4.0,
        se_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.has_skip = (width_in == width_out) and (stride == 1)
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            bn_epsilon,
            bn_momentum,
            activation,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )

        self.depth = self.f.depth

    def forward(self, x):
        return x + self.f(x) if self.has_skip else self.f(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: nn.Module,
        activation: nn.Module,
        group_width: int,
        bottleneck_multiplier: float,
        params: "RegNetParams",
        stage_index: int = 0,
    ):
        super().__init__()
        self.stage_depth = 0

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                params.bn_epsilon,
                params.bn_momentum,
                activation,
                group_width,
                bottleneck_multiplier,
                params.se_ratio,
            )

            self.stage_depth += block.depth
            self.add_module(f"block{stage_index}-{i}", block)


# Now to the RegNet specific part
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


class RegNetParams:
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
class RegNet(ClassyModel):
    """Implementation of RegNet, a particular form of AnyNets.

    See https://arxiv.org/abs/2003.13678 for introduction to RegNets, and details about
    RegNetX and RegNetY models.

    See https://arxiv.org/abs/2103.06877 for details about RegNetZ models.
    """

    def __init__(self, params: RegNetParams):
        super().__init__()

        silu = None if get_torch_version() < [1, 7] else nn.SiLU()
        activation = {
            ActivationType.RELU: nn.ReLU(params.relu_in_place),
            ActivationType.SILU: silu,
        }[params.activation]

        if activation is None:
            raise RuntimeError("SiLU activation is only supported since PyTorch 1.7")

        # Ad hoc stem
        self.stem = {
            StemType.RES_STEM_CIFAR: ResStemCifar,
            StemType.RES_STEM_IN: ResStemIN,
            StemType.SIMPLE_STEM_IN: SimpleStemIN,
        }[params.stem_type](
            3,
            params.stem_width,
            params.bn_epsilon,
            params.bn_momentum,
            activation,
        )

        # Instantiate all the AnyNet blocks in the trunk
        block_fun = {
            BlockType.VANILLA_BLOCK: VanillaBlock,
            BlockType.RES_BASIC_BLOCK: ResBasicBlock,
            BlockType.RES_BOTTLENECK_BLOCK: ResBottleneckBlock,
            BlockType.RES_BOTTLENECK_LINEAR_BLOCK: ResBottleneckLinearBlock,
        }[params.block_type]

        current_width = params.stem_width

        self.trunk_depth = 0

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(params.get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_fun,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        params,
                        stage_index=i + 1,
                    ),
                )
            )

            self.trunk_depth += blocks[-1][1].stage_depth

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        # Init weights and good to go
        self.init_weights()

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
                depth=20, w_0=368, w_a=102.79, w_m=2.05, group_width=496, **kwargs
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
