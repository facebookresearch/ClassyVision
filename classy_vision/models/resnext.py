#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of ResNeXt (https://arxiv.org/pdf/1611.05431.pdf)
"""

import copy
import math
import re
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn
from classy_vision.generic.util import is_pos_int, is_pos_int_tuple

from . import register_model
from .classy_model import ClassyModel
from .squeeze_and_excitation_layer import SqueezeAndExcitationLayer


# version number for the current implementation
VERSION = 0.2
# global setting for in-place ReLU:
INPLACE = True


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """helper function for constructing 3x3 grouped convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """helper function for constructing 1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GenericLayer(nn.Module):
    """
    Parent class for 2-layer (BasicLayer) and 3-layer (BottleneckLayer)
    bottleneck layer class
    """

    def __init__(
        self,
        convolutional_block,
        in_planes,
        out_planes,
        stride=1,
        mid_planes_and_cardinality=None,
        reduction=4,
        final_bn_relu=True,
        use_se=False,
        se_reduction_ratio=16,
    ):

        # assertions on inputs:
        assert is_pos_int(in_planes) and is_pos_int(out_planes)
        assert (is_pos_int(stride) or is_pos_int_tuple(stride)) and is_pos_int(
            reduction
        )

        # set object fields:
        super(GenericLayer, self).__init__()
        self.convolutional_block = convolutional_block
        self.final_bn_relu = final_bn_relu

        # final batchnorm and relu layer:
        if final_bn_relu:
            self.bn = nn.BatchNorm2d(out_planes)
            self.relu = nn.ReLU(inplace=INPLACE)

        # define down-sampling layer (if direct residual impossible):
        self.downsample = None
        if (stride != 1 and stride != (1, 1)) or in_planes != out_planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes, stride=stride),
                nn.BatchNorm2d(out_planes),
            )

        self.se = (
            SqueezeAndExcitationLayer(out_planes, reduction_ratio=se_reduction_ratio)
            if use_se
            else None
        )

    def forward(self, x):

        # if required, perform downsampling along shortcut connection:
        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)

        # forward pass through convolutional block:
        out = self.convolutional_block(x)

        if self.final_bn_relu:
            out = self.bn(out)

        if self.se is not None:
            out = self.se(out)

        # add residual connection, perform rely + batchnorm, and return result:
        out += residual
        if self.final_bn_relu:
            out = self.relu(out)
        return out


class BasicLayer(GenericLayer):
    """
    ResNeXt layer with `in_planes` input planes and `out_planes`
    output planes.
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        mid_planes_and_cardinality=None,
        reduction=1,
        final_bn_relu=True,
        use_se=False,
        se_reduction_ratio=16,
    ):

        # assertions on inputs:
        assert is_pos_int(in_planes) and is_pos_int(out_planes)
        assert (is_pos_int(stride) or is_pos_int_tuple(stride)) and is_pos_int(
            reduction
        )

        # define convolutional block:
        convolutional_block = nn.Sequential(
            conv3x3(in_planes, out_planes, stride=stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=INPLACE),
            conv3x3(out_planes, out_planes),
        )

        # call constructor of generic layer:
        super().__init__(
            convolutional_block,
            in_planes,
            out_planes,
            stride=stride,
            reduction=reduction,
            final_bn_relu=final_bn_relu,
            use_se=use_se,
            se_reduction_ratio=se_reduction_ratio,
        )


class BottleneckLayer(GenericLayer):
    """
    ResNeXt bottleneck layer with `in_planes` input planes, `out_planes`
    output planes, and a bottleneck `reduction`.
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        mid_planes_and_cardinality=None,
        reduction=4,
        final_bn_relu=True,
        use_se=False,
        se_reduction_ratio=16,
    ):

        # assertions on inputs:
        assert is_pos_int(in_planes) and is_pos_int(out_planes)
        assert (is_pos_int(stride) or is_pos_int_tuple(stride)) and is_pos_int(
            reduction
        )

        # define convolutional layers:
        bottleneck_planes = int(math.ceil(out_planes / reduction))
        cardinality = 1
        if mid_planes_and_cardinality is not None:
            mid_planes, cardinality = mid_planes_and_cardinality
            bottleneck_planes = mid_planes * cardinality

        convolutional_block = nn.Sequential(
            conv1x1(in_planes, bottleneck_planes),
            nn.BatchNorm2d(bottleneck_planes),
            nn.ReLU(inplace=INPLACE),
            conv3x3(
                bottleneck_planes, bottleneck_planes, stride=stride, groups=cardinality
            ),
            nn.BatchNorm2d(bottleneck_planes),
            nn.ReLU(inplace=INPLACE),
            conv1x1(bottleneck_planes, out_planes),
        )

        # call constructor of generic layer:
        super(BottleneckLayer, self).__init__(
            convolutional_block,
            in_planes,
            out_planes,
            stride=stride,
            reduction=reduction,
            final_bn_relu=final_bn_relu,
            use_se=use_se,
            se_reduction_ratio=se_reduction_ratio,
        )


class SmallInputInitialBlock(nn.Module):
    """
    ResNeXt initial block for small input with `in_planes` input planes
    """

    def __init__(self, init_planes):
        super().__init__()
        self._module = nn.Sequential(
            conv3x3(3, init_planes, stride=1),
            nn.BatchNorm2d(init_planes),
            nn.ReLU(inplace=INPLACE),
        )

    def forward(self, x):
        return self._module(x)


class InitialBlock(nn.Module):
    """
    ResNeXt initial block with `in_planes` input planes
    """

    def __init__(self, init_planes):
        super().__init__()
        self._module = nn.Sequential(
            nn.Conv2d(3, init_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_planes),
            nn.ReLU(inplace=INPLACE),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self._module(x)


@register_model("resnext")
class ResNeXt(ClassyModel):
    __jit_unused_properties__ = ClassyModel.__jit_unused_properties__ + ["model_depth"]

    def __init__(
        self,
        num_blocks,
        init_planes: int = 64,
        reduction: int = 4,
        small_input: bool = False,
        zero_init_bn_residuals: bool = False,
        base_width_and_cardinality: Optional[Union[Tuple, List]] = None,
        basic_layer: bool = False,
        final_bn_relu: bool = True,
        use_se: bool = False,
        se_reduction_ratio: int = 16,
    ):
        """
        Implementation of `ResNeXt <https://arxiv.org/pdf/1611.05431.pdf>`_.

        Args:
            small_input: set to `True` for 32x32 sized image inputs.
            final_bn_relu: set to `False` to exclude the final batchnorm and
                ReLU layers. These settings are useful when training Siamese
                networks.
            use_se: Enable squeeze and excitation
            se_reduction_ratio: The reduction ratio to apply in the excitation
                stage. Only used if `use_se` is `True`.
        """
        super().__init__()

        # assertions on inputs:
        assert type(num_blocks) == list
        assert all(is_pos_int(n) for n in num_blocks)
        assert is_pos_int(init_planes) and is_pos_int(reduction)
        assert type(small_input) == bool
        assert (
            type(zero_init_bn_residuals) == bool
        ), "zero_init_bn_residuals must be a boolean, set to true if gamma of last\
             BN of residual block should be initialized to 0.0, false for 1.0"
        assert base_width_and_cardinality is None or (
            isinstance(base_width_and_cardinality, (tuple, list))
            and len(base_width_and_cardinality) == 2
            and is_pos_int(base_width_and_cardinality[0])
            and is_pos_int(base_width_and_cardinality[1])
        )
        assert isinstance(use_se, bool), "use_se has to be a boolean"

        # initial convolutional block:
        self.num_blocks = num_blocks
        self.small_input = small_input
        self._make_initial_block(small_input, init_planes, basic_layer)

        # compute number of planes at each spatial resolution:
        out_planes = [init_planes * 2 ** i * reduction for i in range(len(num_blocks))]
        in_planes = [init_planes] + out_planes[:-1]

        # create subnetworks for each spatial resolution:
        blocks = []
        for idx in range(len(out_planes)):
            mid_planes_and_cardinality = None
            if base_width_and_cardinality is not None:
                w, c = base_width_and_cardinality
                mid_planes_and_cardinality = (w * 2 ** idx, c)
            new_block = self._make_resolution_block(
                in_planes[idx],
                out_planes[idx],
                idx,
                num_blocks[idx],  # num layers
                stride=1 if idx == 0 else 2,
                mid_planes_and_cardinality=mid_planes_and_cardinality,
                reduction=reduction,
                final_bn_relu=final_bn_relu or (idx != (len(out_planes) - 1)),
                use_se=use_se,
                se_reduction_ratio=se_reduction_ratio,
            )
            blocks.append(new_block)
        self.blocks = nn.Sequential(*blocks)

        self.out_planes = out_planes[-1]
        self._num_classes = out_planes

        # initialize weights:
        self._initialize_weights(zero_init_bn_residuals)

    def _initialize_weights(self, zero_init_bn_residuals):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Init BatchNorm gamma to 0.0 for last BN layer, it gets 0.2-0.3% higher
        # final val top1 for larger batch sizes.
        if zero_init_bn_residuals:
            for m in self.modules():
                if isinstance(m, GenericLayer):
                    if hasattr(m, "bn"):
                        nn.init.constant_(m.bn.weight, 0)

    def _make_initial_block(self, small_input, init_planes, basic_layer):
        if small_input:
            self.initial_block = SmallInputInitialBlock(init_planes)
            self.layer_type = BasicLayer
        else:
            self.initial_block = InitialBlock(init_planes)
            self.layer_type = BasicLayer if basic_layer else BottleneckLayer

    # helper function that creates ResNet blocks at single spatial resolution:
    def _make_resolution_block(
        self,
        in_planes,
        out_planes,
        resolution_idx,
        num_blocks,
        stride=1,
        mid_planes_and_cardinality=None,
        reduction=4,
        final_bn_relu=True,
        use_se=False,
        se_reduction_ratio=16,
    ):
        # add the desired number of residual blocks:
        blocks = OrderedDict()
        for idx in range(num_blocks):
            block_name = "block{}-{}".format(resolution_idx, idx)
            blocks[block_name] = self.layer_type(
                in_planes if idx == 0 else out_planes,
                out_planes,
                stride=stride if idx == 0 else 1,  # only first block has stride
                mid_planes_and_cardinality=mid_planes_and_cardinality,
                reduction=reduction,
                final_bn_relu=final_bn_relu or (idx != (num_blocks - 1)),
                use_se=use_se,
                se_reduction_ratio=se_reduction_ratio,
            )
        return nn.Sequential(blocks)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResNeXt":
        """Instantiates a ResNeXt from a configuration.

        Args:
            config: A configuration for a ResNeXt.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A ResNeXt instance.
        """
        assert "num_blocks" in config

        basic_layer = config.get("basic_layer", False)
        config = {
            "num_blocks": config["num_blocks"],
            "init_planes": config.get("init_planes", 64),
            "reduction": config.get("reduction", 1 if basic_layer else 4),
            "base_width_and_cardinality": config.get("base_width_and_cardinality"),
            "small_input": config.get("small_input", False),
            "basic_layer": basic_layer,
            "final_bn_relu": config.get("final_bn_relu", True),
            "zero_init_bn_residuals": config.get("zero_init_bn_residuals", False),
            "use_se": config.get("use_se", False),
            "se_reduction_ratio": config.get("se_reduction_ratio", 16),
        }
        return cls(**config)

    # forward pass in residual network:
    def forward(self, x):
        # initial convolutional block:
        out = self.initial_block(x)

        # evaluate all residual blocks:
        # TODO: (kaizh) T43794289 exit early if there is no block that has heads
        out = self.blocks(out)

        return out

    def _convert_model_state(self, state):
        """Convert model state from the old implementation to the current format.

        Updates the state dict in place and returns True if the state dict was updated.
        """
        pattern = r"blocks\.(?P<block_id_0>[0-9]*)\.(?P<block_id_1>[0-9]*)\._module\."
        repl = r"blocks.\g<block_id_0>.block\g<block_id_0>-\g<block_id_1>."
        trunk_dict = state["model"]["trunk"]
        new_trunk_dict = {}
        replaced_keys = False
        for key, value in trunk_dict.items():
            new_key = re.sub(pattern, repl, key)
            if new_key != key:
                replaced_keys = True
            new_trunk_dict[new_key] = value
        state["model"]["trunk"] = new_trunk_dict
        state["version"] = VERSION
        return replaced_keys

    def get_classy_state(self, deep_copy=False):
        state = super().get_classy_state(deep_copy=deep_copy)
        state["version"] = VERSION
        return state

    def set_classy_state(self, state, strict=True):
        version = state.get("version")
        if version is None:
            # convert the weights from the previous implementation of ResNeXt to the
            # current one
            if not self._convert_model_state(state):
                raise RuntimeError("ResNeXt state conversion failed")
            message = (
                "Provided state dict is from an old implementation of ResNeXt. "
                "This has been deprecated and will be removed soon."
            )
            warnings.warn(message, DeprecationWarning, stacklevel=2)
        elif version != VERSION:
            raise ValueError(
                f"Unsupported ResNeXt version: {version}. Expected: {VERSION}"
            )
        super().set_classy_state(state, strict)


class _ResNeXt(ResNeXt):
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResNeXt":
        config = copy.deepcopy(config)
        config.pop("name")
        if "heads" in config:
            config.pop("heads")
        return cls(**config)


@register_model("resnet18")
class ResNet18(_ResNeXt):
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[2, 2, 2, 2],
            basic_layer=True,
            zero_init_bn_residuals=True,
            reduction=1,
            **kwargs,
        )


@register_model("resnet34")
class ResNet34(_ResNeXt):
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 6, 3],
            basic_layer=True,
            zero_init_bn_residuals=True,
            reduction=1,
            **kwargs,
        )


@register_model("resnet50")
class ResNet50(_ResNeXt):
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 6, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            **kwargs,
        )


@register_model("resnet101")
class ResNet101(_ResNeXt):
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 23, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            **kwargs,
        )


@register_model("resnet152")
class ResNet152(_ResNeXt):
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 8, 36, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            **kwargs,
        )


# Note, the ResNeXt models all have weight decay enabled for the batch
# norm parameters. We have found empirically that this gives better
# results when training on ImageNet (~0.5pp of top-1 acc) and brings
# our results on track with reported ImageNet results...but for
# training on other datasets, we have observed losses in accuracy (for
# example, the dataset used in https://arxiv.org/abs/1805.00932).
@register_model("resnext50_32x4d")
class ResNeXt50(_ResNeXt):
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 6, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            base_width_and_cardinality=(4, 32),
            **kwargs,
        )


@register_model("resnext101_32x4d")
class ResNeXt101(_ResNeXt):
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 23, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            base_width_and_cardinality=(4, 32),
            **kwargs,
        )


@register_model("resnext152_32x4d")
class ResNeXt152(_ResNeXt):
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 8, 36, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            base_width_and_cardinality=(4, 32),
            **kwargs,
        )
