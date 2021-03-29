#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict
from enum import Enum, auto
from typing import Any, Dict, Optional, Sequence

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
        params: "AnyNetParams",
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


class AnyNetParams:
    def __init__(
        self,
        depths: Sequence[int],
        widths: Sequence[int],
        group_widths: Sequence[int],
        bottleneck_multipliers: Sequence[int],
        strides: Sequence[int],
        stem_type: StemType = StemType.SIMPLE_STEM_IN,
        stem_width: int = 32,
        block_type: BlockType = BlockType.RES_BOTTLENECK_BLOCK,
        activation: ActivationType = ActivationType.RELU,
        use_se: bool = True,
        se_ratio: float = 0.25,
        bn_epsilon: float = 1e-05,
        bn_momentum: bool = 0.1,
    ):
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.stem_type = stem_type
        self.stem_width = stem_width
        self.block_type = block_type
        self.activation = activation
        self.use_se = use_se
        self.se_ratio = se_ratio if use_se else None
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.relu_in_place = RELU_IN_PLACE

    def get_expanded_params(self):
        """Return an iterator over AnyNet parameters for each stage."""
        return zip(
            self.widths,
            self.strides,
            self.depths,
            self.group_widths,
            self.bottleneck_multipliers,
        )


@register_model("anynet")
class AnyNet(ClassyModel):
    """Implementation of an AnyNet.

    See https://arxiv.org/abs/2003.13678 for details.
    """

    def __init__(self, params: AnyNetParams):
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
    def from_config(cls, config: Dict[str, Any]) -> "AnyNet":
        """Instantiates an AnyNet from a configuration.

        Args:
            config: A configuration for an AnyNet.
                See `AnyNetParams` for parameters expected in the config.

        Returns:
            An AnyNet instance.
        """

        params = AnyNetParams(
            depths=config["depths"],
            widths=config["widths"],
            group_widths=config["group_widths"],
            bottleneck_multipliers=config["bottleneck_multipliers"],
            strides=config["strides"],
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
