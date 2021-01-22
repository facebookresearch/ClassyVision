#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from collections import OrderedDict
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from classy_vision.models import ClassyModel, register_model
from torch.nn import functional as F


class BlockParams(NamedTuple):
    num_repeat: int
    kernel_size: int
    stride: int
    expand_ratio: float
    input_filters: int
    output_filters: int
    se_ratio: float
    id_skip: bool


class EfficientNetParams(NamedTuple):
    width_coefficient: float
    depth_coefficient: float
    resolution: int
    dropout_rate: float


BLOCK_PARAMS = [
    BlockParams(1, 3, 1, 1, 32, 16, 0.25, True),
    BlockParams(2, 3, 2, 6, 16, 24, 0.25, True),
    BlockParams(2, 5, 2, 6, 24, 40, 0.25, True),
    BlockParams(3, 3, 2, 6, 40, 80, 0.25, True),
    BlockParams(3, 5, 1, 6, 80, 112, 0.25, True),
    BlockParams(4, 5, 2, 6, 112, 192, 0.25, True),
    BlockParams(1, 3, 1, 6, 192, 320, 0.25, True),
]


MODEL_PARAMS = {
    "B0": EfficientNetParams(1.0, 1.0, 224, 0.2),
    "B1": EfficientNetParams(1.0, 1.1, 240, 0.2),
    "B2": EfficientNetParams(1.1, 1.2, 260, 0.3),
    "B3": EfficientNetParams(1.2, 1.4, 300, 0.3),
    "B4": EfficientNetParams(1.4, 1.8, 380, 0.4),
    "B5": EfficientNetParams(1.6, 2.2, 456, 0.4),
    "B6": EfficientNetParams(1.8, 2.6, 528, 0.5),
    "B7": EfficientNetParams(2.0, 3.1, 600, 0.5),
}


def swish(x):
    """
    Swish activation function.
    """
    return x * torch.sigmoid(x)


def drop_connect(inputs, is_training, drop_connect_rate):
    """
    Apply drop connect to random inputs in a batch.
    """
    if not is_training:
        return inputs

    keep_prob = 1 - drop_connect_rate

    # compute drop connect tensor
    batch_size = inputs.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    outputs = (inputs / keep_prob) * binary_tensor
    return outputs


def scale_width(num_filters, width_coefficient, width_divisor, min_width):
    """
    Calculates the scaled number of filters based on the width coefficient and
    rounds the result by the width divisor.
    """
    if not width_coefficient:
        return num_filters

    num_filters *= width_coefficient
    min_width = min_width or width_divisor
    new_filters = max(
        min_width,
        (int(num_filters + width_divisor / 2) // width_divisor) * width_divisor,
    )
    # Do not round down by more than 10%
    if new_filters < 0.9 * num_filters:
        new_filters += width_divisor
    return int(new_filters)


def scale_depth(num_repeats, depth_coefficient):
    """
    Calculates the scaled number of repeats based on the depth coefficient.
    """
    if not depth_coefficient:
        return num_repeats
    return int(math.ceil(depth_coefficient * num_repeats))


def calculate_output_image_size(input_image_size, stride):
    """
    Calculates the output image size when using Conv2dSamePadding with a stride
    """
    image_height, image_width = input_image_size
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return image_height, image_width


class Conv2dSamePadding(nn.Conv2d):
    """
    Conv2d, but with 'same' convolutions like TensorFlow.
    """

    def __init__(
        self, image_size, in_channels, out_channels, kernel_size, **kernel_wargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kernel_wargs)

        image_h, image_w = image_size
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        out_h, out_w = math.ceil(image_h / stride_h), math.ceil(image_w / stride_w)
        pad_h = max(
            (out_h - 1) * self.stride[0] + (kernel_h - 1) * dilation_h + 1 - image_h, 0
        )
        pad_w = max(
            (out_w - 1) * self.stride[1] + (kernel_w - 1) * dilation_w + 1 - image_w, 0
        )
        self.out_h = out_h
        self.out_w = out_w
        self.same_padding = None
        if pad_h > 0 or pad_w > 0:
            self.same_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        self.image_size = image_size

    def forward(self, x):
        input_image_size = x.shape[-2:]
        assert (
            input_image_size == self.image_size
        ), f"Input shape mismatch, got: {input_image_size}, expected: {self.image_size}"
        if self.same_padding is not None:
            x = self.same_padding(x)
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x

    def flops(self, x):
        batchsize_per_replica = x.size()[0]
        return (
            batchsize_per_replica
            * self.in_channels
            * self.out_channels
            * self.kernel_size[0]
            * self.kernel_size[1]
            * self.out_h
            * self.out_w
            / self.groups
        )

    def activations(self, x, out):
        return out.numel()


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        input_filters: int,
        output_filters: int,
        expand_ratio: float,
        kernel_size: int,
        stride: int,
        se_ratio: float,
        id_skip: bool,
        use_se: bool,
        bn_momentum: float,
        bn_epsilon: float,
    ):
        assert se_ratio is None or (0 < se_ratio <= 1)
        super().__init__()
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.has_se = use_se and se_ratio is not None
        self.se_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.id_skip = id_skip
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.input_filters = input_filters
        self.output_filters = output_filters

        self.relu_fn = swish

        # used to track the depth of the block
        self.depth = 0

        # Expansion phase
        expanded_filters = input_filters * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = Conv2dSamePadding(
                image_size=image_size,
                in_channels=input_filters,
                out_channels=expanded_filters,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            self.bn0 = nn.BatchNorm2d(
                num_features=expanded_filters,
                momentum=self.bn_momentum,
                eps=self.bn_epsilon,
            )
            self.depth += 1

        # Depthwise convolution phase
        self.depthwise_conv = Conv2dSamePadding(
            image_size=image_size,
            in_channels=expanded_filters,
            out_channels=expanded_filters,
            groups=expanded_filters,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=expanded_filters,
            momentum=self.bn_momentum,
            eps=self.bn_epsilon,
        )
        self.depth += 1

        image_size = calculate_output_image_size(image_size, stride)

        if self.has_se:
            # Squeeze and Excitation layer
            num_reduced_filters = max(1, int(input_filters * se_ratio))
            self.se_reduce = Conv2dSamePadding(
                image_size=(1, 1),
                in_channels=expanded_filters,
                out_channels=num_reduced_filters,
                kernel_size=1,
                stride=1,
                bias=True,
            )
            self.se_expand = Conv2dSamePadding(
                image_size=(1, 1),
                in_channels=num_reduced_filters,
                out_channels=expanded_filters,
                kernel_size=1,
                stride=1,
                bias=True,
            )
            self.depth += 2

        # Output phase
        self.project_conv = Conv2dSamePadding(
            image_size=image_size,
            in_channels=expanded_filters,
            out_channels=output_filters,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=output_filters, momentum=self.bn_momentum, eps=self.bn_epsilon
        )
        self.depth += 1

    def forward(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        if self.expand_ratio != 1:
            x = self.relu_fn(self.bn0(self.expand_conv(inputs)))
        else:
            x = inputs

        x = self.relu_fn(self.bn1(self.depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            # squeeze x in the spatial dimensions
            x_squeezed = self.se_avgpool(x)
            x_expanded = self.se_expand(self.relu_fn(self.se_reduce(x_squeezed)))
            x = torch.sigmoid(x_expanded) * x

        x = self.bn2(self.project_conv(x))

        # Skip connection and Drop Connect
        if self.id_skip:
            if self.stride == 1 and self.input_filters == self.output_filters:
                # only apply drop connect if a skip connection is present
                if drop_connect_rate:
                    x = drop_connect(x, self.training, drop_connect_rate)
                x = x + inputs
        return x


@register_model("efficientnet")
class EfficientNet(ClassyModel):
    """
    Implementation of EfficientNet, https://arxiv.org/pdf/1905.11946.pdf
    References:
        https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
        https://github.com/lukemelas/EfficientNet-PyTorch

    NOTE: the original implementation uses the names depth_divisor and min_depth
          to refer to the number of channels, which is confusing, since the paper
          refers to the channel dimension as width. We use the width_divisor and
          min_width names instead.
    """

    def __init__(
        self,
        num_classes: int,
        model_params: EfficientNetParams,
        bn_momentum: float,
        bn_epsilon: float,
        width_divisor: int,
        min_width: Optional[int],
        drop_connect_rate: float,
        use_se: bool,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.image_resolution = model_params.resolution

        # use the swish activation function
        self.relu_fn = swish

        # width and depth parameters
        width_coefficient = model_params.width_coefficient
        depth_coefficient = model_params.depth_coefficient

        # drop connect rate
        self.drop_connect_rate = drop_connect_rate

        # input dimensions
        in_channels = self.input_shape[0]
        image_size = self.input_shape[1:]

        # Stem
        out_channels = 32
        out_channels = scale_width(
            out_channels, width_coefficient, width_divisor, min_width
        )
        self.conv_stem = Conv2dSamePadding(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_momentum, eps=bn_epsilon
        )
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        blocks = OrderedDict()
        for block_idx, block_params in enumerate(BLOCK_PARAMS):
            assert block_params.num_repeat > 0, "num_repeat has to be > 0"

            # Update block input and output filters based on the width_coefficient,
            # and the number of repeats based on the depth_coefficient
            block_params = block_params._replace(
                input_filters=scale_width(
                    block_params.input_filters,
                    width_coefficient,
                    width_divisor,
                    min_width,
                ),
                output_filters=scale_width(
                    block_params.output_filters,
                    width_coefficient,
                    width_divisor,
                    min_width,
                ),
                num_repeat=scale_depth(block_params.num_repeat, depth_coefficient),
            )

            block_name = f"block{block_idx}-0"
            # The first block needs to take care of the stride and filter size increase
            blocks[block_name] = MBConvBlock(
                image_size,
                block_params.input_filters,
                block_params.output_filters,
                block_params.expand_ratio,
                block_params.kernel_size,
                block_params.stride,
                block_params.se_ratio,
                block_params.id_skip,
                use_se,
                bn_momentum,
                bn_epsilon,
            )
            image_size = calculate_output_image_size(image_size, block_params.stride)
            if block_params.num_repeat > 1:
                block_params = block_params._replace(
                    input_filters=block_params.output_filters, stride=1
                )
            for i in range(1, block_params.num_repeat):
                block_name = f"block{block_idx}-{i}"
                blocks[block_name] = MBConvBlock(
                    image_size,
                    block_params.input_filters,
                    block_params.output_filters,
                    block_params.expand_ratio,
                    block_params.kernel_size,
                    block_params.stride,
                    block_params.se_ratio,
                    block_params.id_skip,
                    use_se,
                    bn_momentum,
                    bn_epsilon,
                )
        self.blocks = nn.Sequential(blocks)

        # Head
        in_channels = block_params.output_filters
        out_channels = 1280
        out_channels = scale_width(
            out_channels, width_coefficient, width_divisor, min_width
        )
        self.conv_head = Conv2dSamePadding(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_momentum, eps=bn_epsilon
        )

        # add a trunk_output module to attach heads to
        self.trunk_output = nn.Identity()

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(out_channels, num_classes)

        if model_params.dropout_rate > 0:
            self.dropout = nn.Dropout(p=model_params.dropout_rate)
        else:
            self.dropout = None

        # initialize weights
        self.init_weights()

    @classmethod
    def from_config(cls, config):
        """Instantiates an EfficientNet from a configuration.

        Args:
            config: A configuration for an EfficientNet.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A ResNeXt instance.
        """
        config = copy.deepcopy(config)
        del config["name"]
        if "heads" in config:
            del config["heads"]
        if "model_name" in config:
            assert (
                config["model_name"] in MODEL_PARAMS
            ), f"Unknown model_name: {config['model_name']}"
            model_params = MODEL_PARAMS[config["model_name"]]
            del config["model_name"]
        else:
            assert "model_params" in config, "Need either model_name or model_params"
            model_params = EfficientNetParams(**config["model_params"])
        config["model_params"] = model_params
        return cls(**config)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                kernel_height, kernel_width = module.kernel_size
                out_channels = module.out_channels
                fan_out = kernel_height * kernel_width * out_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init_range = 1.0 / math.sqrt(module.out_features)
                nn.init.uniform_(module.weight, -init_range, init_range)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        input_shape = inputs.shape[1:]
        assert (
            input_shape == self.input_shape
        ), f"Input shape mismatch, got: {input_shape}, expected: {self.input_shape}"

        # Stem
        outputs = self.relu_fn(self.bn0(self.conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.blocks)
            outputs = block(outputs, drop_connect_rate=drop_connect_rate)

        # Conv head
        outputs = self.relu_fn(self.bn1(self.conv_head(outputs)))

        # Trunk output (identity function)
        outputs = self.trunk_output(outputs)

        # Average Pooling
        outputs = self.avg_pooling(outputs).view(outputs.size(0), -1)

        # Dropout
        if self.dropout is not None:
            outputs = self.dropout(outputs)

        # Fully connected layer
        outputs = self.fc(outputs)
        return outputs

    @property
    def input_shape(self):
        return (3, self.image_resolution, self.image_resolution)


class _EfficientNet(EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(
            bn_momentum=0.01,
            bn_epsilon=1e-3,
            drop_connect_rate=0.2,
            num_classes=1000,
            width_divisor=8,
            min_width=None,
            use_se=True,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EfficientNet":
        config = copy.deepcopy(config)
        config.pop("name")
        if "heads" in config:
            config.pop("heads")
        return cls(**config)


@register_model("efficientnet_b0")
class EfficientNetB0(_EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS["B0"])


@register_model("efficientnet_b1")
class EfficientNetB1(_EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS["B1"])


@register_model("efficientnet_b2")
class EfficientNetB2(_EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS["B2"])


@register_model("efficientnet_b3")
class EfficientNetB3(_EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS["B3"])


@register_model("efficientnet_b4")
class EfficientNetB4(_EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS["B4"])


@register_model("efficientnet_b5")
class EfficientNetB5(_EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS["B5"])


@register_model("efficientnet_b6")
class EfficientNetB6(_EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS["B6"])


@register_model("efficientnet_b7")
class EfficientNetB7(_EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS["B7"])
