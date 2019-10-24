#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# dependencies:
import math

import torch
import torch.nn as nn
from classy_vision.generic.util import is_pos_int

from . import register_model
from .classy_model import ClassyModel


# FIXME: set threshold based on computational budget:
THRESHOLD = 1.0


# closure returning single convolutional layer:
def get_preactivation_layer(in_planes, intermediate, out_planes, stride=1):
    """Returns a single convolutional block.

    The convolutional block has `in_planes` input channels, `intermediate`
    intermediate channels, and `out_planes` output channels. The `stride` is
    used in the 3x3 filter.
    """
    return nn.Sequential(
        nn.BatchNorm2d(in_planes),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_planes, intermediate, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(intermediate),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            intermediate,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
    )


class _Classifier(nn.Module):
    """Classifier with `num_planes` input channels and `num_classes classes."""

    def __init__(self, num_planes, num_classes):
        super(_Classifier, self).__init__()
        assert is_pos_int(num_planes), "num_planes must be a positive integer"
        assert is_pos_int(num_classes), "num_classes must be a positive integer"
        self.features = get_preactivation_layer(num_planes, num_planes, num_planes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_planes, num_classes)
        self.num_planes = num_planes
        self.num_classes = num_classes

    def forward(self, x):
        if torch.is_tensor(x):
            x = [x]
        out = self.features(x[-1])
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class _MultiscaleDenseLayer(nn.Module):
    """Single layer of a multi-scale DenseNet.

    The layer contains `in_planes` input channels per scale (list input). The
    number of new channels per layer is given by `growth_rate`, and `bottleneck`
    is a multiplication factor determining the width of intermediate layers.
    The `reduction` argument specifies the reduction factor on of feature map
    size between scales.
    """

    def __init__(self, in_planes, growth_rate=32, bottleneck=4, reduction=2):
        # TODO: Some batchnorm operations are superfluous and can be removed.

        # set input and output size of layer:
        super(_MultiscaleDenseLayer, self).__init__()
        if isinstance(in_planes, int):
            in_planes = [in_planes]
        self.num_scales = len(in_planes)
        self.input_size = in_planes
        self.output_size = [None] * self.num_scales

        # create all convolutional layers:
        self.regular_layers, self.strided_layers = [], []
        prev_scale_strided_growth = 0
        for idx in range(self.num_scales):

            # construct regular convolutional layer:
            reg_growth = (2 ** idx) * growth_rate
            strided_growth = (2 ** (idx + 1)) * growth_rate
            self.regular_layers.append(
                get_preactivation_layer(
                    self.input_size[idx], bottleneck * reg_growth, reg_growth, stride=1
                )
            )

            # construct strided convolutional layer:
            if idx < self.num_scales - 1:
                self.strided_layers.append(
                    get_preactivation_layer(
                        self.input_size[idx],
                        bottleneck * strided_growth,
                        strided_growth,
                        stride=reduction,
                    )
                )

            self.output_size[idx] = (
                self.input_size[idx] + reg_growth + prev_scale_strided_growth
            )
            prev_scale_strided_growth = strided_growth

        # make sure operators are broadcasted correctly:
        self.regular_layers = nn.ModuleList(self.regular_layers)
        self.strided_layers = nn.ModuleList(self.strided_layers)

    def forward(self, x):
        assert isinstance(x, list) or isinstance(x, tuple)
        output = [None] * self.num_scales
        for scale_idx in range(self.num_scales):
            features = self.regular_layers[scale_idx](x[scale_idx])
            if scale_idx > 0:
                features = torch.cat(
                    [features, self.strided_layers[scale_idx - 1](x[scale_idx - 1])], 1
                )
            output[scale_idx] = torch.cat([x[scale_idx], features], 1)
        return tuple(output)


class _MultiscaleInitLayer(nn.Module):
    """Initial layer constructing a multi-scale from a single-scale representation.

    The layer receives `in_planes` input channels (integer), and constructs
    `num_scales` scales for these channels. The `reduction` argument specifies
    the reduction factor on of feature map size between scales.
    """

    def __init__(self, num_scales, in_planes, reduction=2):

        # set input size, size of each scale, and output size of layer:
        super(_MultiscaleInitLayer, self).__init__()
        self.input_size = [in_planes]
        self.scale_size = [in_planes] + [
            in_planes * (2 ** idx) for idx in range(num_scales - 1)
        ]
        self.output_size = [in_planes * (2 ** idx) for idx in range(num_scales)]

        # add all strided convolutional layers:
        self.layers = nn.ModuleList(
            [
                get_preactivation_layer(
                    self.scale_size[idx],
                    self.scale_size[idx],
                    self.output_size[idx],
                    stride=reduction if idx != 0 else 1,
                )
                for idx in range(num_scales)
            ]
        )

    def forward(self, x):
        output = []
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                output.append(layer(x))
            else:
                output.append(layer(output[idx - 1]))

        return output


class _MultiscaleDenseBlock(nn.Module):
    """Block of multi-scale, densely connected layers.

    The block contains `in_planes` input channels per scale (list input), and
    constructs `num_layers` densely connected layers. Classifiers are attached
    at the indices in `classifier_layers` list and have `num_classes` classes.

    The number of new channels per layer is given by `growth_rate`, and
    `bottleneck` is a multiplication factor determining the relative width of
    intermediate layers.
    """

    def __init__(
        self,
        in_planes,
        num_layers,
        num_classes=None,
        classifier_layers=None,
        growth_rate=32,
        bottleneck=4,
    ):

        # initialize instance variables:
        super(_MultiscaleDenseBlock, self).__init__()
        self.num_classes, self.layers, self.classifiers = num_classes, [], []
        self.classifier_layers = (
            classifier_layers if classifier_layers is not None else []
        )

        # set input size of block:
        if isinstance(in_planes, int):
            in_planes = [in_planes]
        self.input_size = in_planes

        # create block of dense layers at same scales:
        for idx in range(num_layers):

            # add layer:
            layer = _MultiscaleDenseLayer(
                in_planes, growth_rate=growth_rate, bottleneck=bottleneck
            )
            self.layers.append(layer)
            in_planes = layer.output_size

            # add classifier:
            if self.num_classes is not None and idx in self.classifier_layers:
                classifier = _Classifier(layer.output_size[-1], num_classes)
                self.classifiers.append(classifier)

        # set output size of block:
        self.output_size = layer.output_size

        # ensure all operators are broadcasted correctly:
        self.layers = nn.ModuleList(self.layers)
        self.classifiers = nn.ModuleList(self.classifiers)
        self.softmax = nn.Softmax(dim=1)  # used at inference time

    def forward(self, x):
        assert isinstance(x, list) or isinstance(x, tuple)
        predictions, classifier_idx = [], 0
        for idx, layer in enumerate(self.layers):

            # compute new feature representation:
            x = layer(x)

            # evaluate classifier:
            if self.num_classes is not None and idx in self.classifier_layers:
                predictions.append(self.classifiers[classifier_idx](x[-1]))
                classifier_idx += 1

                # in inference mode, stop predicting if predictions are certain:
                if not self.training:
                    cur_predictions = predictions[-1].detach()
                    max_softmax, class_idx = self.softmax(cur_predictions).max(1)
                    if max_softmax.gt(THRESHOLD).all().item() == 1:
                        return None, predictions
                        # NOTE: This works on batch-level. Use batchsize_per_replica = 1!  # noqa
        return x, predictions


class _Transition(nn.Module):
    """Transition layer that removes the highest spatial resolution.

    The transition layer receives `in_planes` input channels per scale (list
    input) and reduces it to `len(in_planes) - 1` scales. The `reduction` argument
    specifies the reduction factor of the feature map size between scales. The
    number of planes at each scale is reduced by a factor of `planes_reduction`.
    """

    def __init__(self, in_planes, reduction=2, planes_reduction=2):

        # set input and output size:
        super(_Transition, self).__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        assert len(in_planes) > 1
        self.input_size = in_planes
        self.output_size = [in_planes[idx] for idx in range(1, len(in_planes))]
        self.output_size[0] += in_planes[0]
        self.output_size = [val // planes_reduction for val in self.output_size]

        # create transition layers:
        self.transition = []
        for idx in range(len(self.input_size)):
            self.transition.append(
                get_preactivation_layer(
                    self.input_size[idx],
                    self.input_size[idx],
                    self.input_size[idx] // planes_reduction,
                    stride=reduction if idx == 0 else 1,
                )
            )
        self.transition = nn.ModuleList(self.transition)  # register operators

        # create block that mixes two finest scales:
        combined_size = (self.input_size[0] + self.input_size[1]) // planes_reduction
        self.mixing = nn.Sequential(
            nn.BatchNorm2d(combined_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                combined_size, self.output_size[0], kernel_size=1, stride=1, bias=False
            ),
        )

    def forward(self, x):
        assert isinstance(x, list) or isinstance(x, tuple)
        out = tuple(trans(x[idx]) for idx, trans in enumerate(self.transition))
        mixed = self.mixing(torch.cat([out[0], out[1]], dim=1))
        return (mixed,) + out[2:]


@register_model("msdnet")
class MSDNet(ClassyModel):
    def __init__(
        self,
        num_classes,
        num_blocks,
        init_planes,
        growth_rate,
        bottleneck,
        planes_reduction,
        reduction,
        num_channels,
        small_input,
    ):
        """Multi-scale densely connected network (MSDNet).

        The list `num_blocks` is a list specifying the structure of the network:
        at the end of each block, a transition layer removes the largest scale
        in the network. The number of initial scales is `len(num_blocks)`. The
        number of feature maps added per layer is `growth_rate`. The `bottleneck`
        is a multiplication factor determining the width of intermediate layers.
        The number of planes is reduced by a factor of `planes_reduction` in each
        transition layer. The `reduction` argument specifies the reduction factor
        on of feature map size between scales.

        Set `small_input` to `True` for 32x32 sized image inputs. Otherwise, the
        model will assume 224x224 inputs. The argument `num_channels` sets the
        number of input channels in the image.
        """
        super().__init__(num_classes)

        # assertions:
        assert num_classes is None or is_pos_int(num_classes)
        assert type(num_blocks) == list
        assert all(is_pos_int(b) for b in num_blocks)
        assert is_pos_int(init_planes)
        assert is_pos_int(growth_rate)
        assert is_pos_int(bottleneck)
        assert is_pos_int(planes_reduction)
        assert is_pos_int(reduction)
        assert is_pos_int(num_channels)
        assert type(small_input) == bool

        # Construct classifier layers
        classifier_layers, layer_idx, offset = [], 1, 1
        while layer_idx < sum(num_blocks):
            classifier_layers.append(layer_idx)
            layer_idx += offset
            offset += 1

        num_scales = len(num_blocks)

        # initial convolutional block:
        self.num_blocks = num_blocks
        if small_input:
            self.initial_block = nn.Sequential(
                nn.Conv2d(
                    num_channels,
                    init_planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(init_planes),
                nn.ReLU(inplace=True),
            )
        else:
            self.initial_block = nn.Sequential(
                nn.Conv2d(
                    num_channels,
                    init_planes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm2d(init_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        # block that creates multi-scale representation:
        self.transitions, self.blocks = [], []
        self.transitions.append(
            _MultiscaleInitLayer(num_scales, init_planes, reduction=reduction)
        )

        # loop over spatial resolutions:
        layer_idx = 0
        for idx, num_layers in enumerate(num_blocks):

            # add dense block:
            cur_classifier_layers = [(c - layer_idx) for c in classifier_layers]
            block = _MultiscaleDenseBlock(
                self.transitions[-1].output_size,
                num_layers,
                num_classes=num_classes,
                classifier_layers=cur_classifier_layers,
                growth_rate=growth_rate,
                bottleneck=bottleneck,
            )
            self.blocks.append(block)
            layer_idx += num_layers

            # add transition layer:
            if idx != len(num_blocks) - 1:
                transition = _Transition(
                    self.blocks[-1].output_size, planes_reduction=2, reduction=reduction
                )
                self.transitions.append(transition)

        # final classifier:
        assert len(self.blocks[-1].output_size) == 1
        num_planes = self.blocks[-1].output_size[-1]
        self.fc = (
            _Classifier(num_planes, num_classes) if num_classes is not None else None
        )

        # ensure all operators are broadcasted correctly:
        self.blocks = nn.ModuleList(self.blocks)
        self.transitions = nn.ModuleList(self.transitions)

        # initialize weights of convolutional and batchnorm layers:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0.0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def update_classifier(self, num_classes):
        """
        When the pre-training dataset has a different number of classes
        than the final data than the final data, this function can be
        used to adjust the classifier outputs.  Note, the _Classifier
        method update_classifier_settings(num_classes) reset the
        classifier if the classes are the same.
        """
        assert num_classes is None or is_pos_int(num_classes)

        if num_classes:
            for block in self.blocks:
                block.num_classes = num_classes
                for classifier in block.classifiers:
                    classifier.update_classifier(num_classes)
            self.fc.update_classifier(num_classes)
        else:
            for block in self.blocks:
                block.num_classes = None
                block.classifiers = []
            self.fc = None

        return self

    @classmethod
    def from_config(cls, config):
        assert "num_blocks" in config
        config = {
            "num_blocks": config["num_blocks"],
            "num_classes": config.get("num_classes"),
            "init_planes": config.get("init_planes", 64),
            "growth_rate": config.get("growth_rate", 32),
            "bottleneck": config.get("bottleneck", 4),
            "planes_reduction": config.get("planes_reduction", 2),
            "reduction": config.get("reduction", 2),
            "num_channels": config.get("num_channels", 3),
            "small_input": config.get("small_input", False),
        }
        return cls(**config)

    # forward pass in DenseNet:
    def forward(self, x):

        # evaluate all the densely connected blocks:
        predictions = []
        x = self.initial_block(x)
        for idx in range(len(self.blocks)):
            x = self.transitions[idx](x)
            x, _predictions = self.blocks[idx](x)
            # _predictions expected to be a list of tensors
            predictions.extend(_predictions)
            if x is None:  # early exit
                # TODO: T41726145
                # Currently classy trainer does not accept lists of tensors.
                return predictions

        # return predictions or features:
        if self.fc is not None:
            predictions.append(self.fc(x))
            # TODO: T41726145
            # Currently classy trainer does not accept lists of tensors.
            return predictions
        else:
            return x  # no classifiers; perform feature extraction

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
        return sum(self.num_blocks)
