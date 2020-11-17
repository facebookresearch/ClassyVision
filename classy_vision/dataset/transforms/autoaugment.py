#!/usr/bin/env python3
# Portions Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# MIT License
#
# Copyright (c) 2018 Philip Popien
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Code modified from
# https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py

import random
import random
from enum import Enum, auto
from functools import partial
from typing import Any
from typing import Tuple, Any, NamedTuple, Sequence, Callable

import numpy as np
from classy_vision.dataset.transforms import ClassyTransform, register_transform
from PIL import Image, ImageEnhance, ImageOps


MIDDLE_GRAY = (128, 128, 128)


class ImageOp(Enum):
    SHEAR_X = auto()
    SHEAR_Y = auto()
    TRANSLATE_X = auto()
    TRANSLATE_Y = auto()
    ROTATE = auto()
    AUTO_CONTRAST = auto()
    INVERT = auto()
    EQUALIZE = auto()
    SOLARIZE = auto()
    POSTERIZE = auto()
    CONTRAST = auto()
    COLOR = auto()
    BRIGHTNESS = auto()
    SHARPNESS = auto()


class ImageOpSetting(NamedTuple):
    ranges: Sequence
    function: Callable


def shear_x(img: Any, magnitude: int, fillcolor: Any = None) -> Any:
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        Image.BICUBIC,
        fillcolor=fillcolor,
    )


def shear_y(img: Any, magnitude: int, fillcolor: Any = None) -> Any:
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        Image.BICUBIC,
        fillcolor=fillcolor,
    )


def translate_x(img: Any, magnitude: int, fillcolor: Any = None) -> Any:
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
        fillcolor=fillcolor,
    )


def translate_y(img: Any, magnitude: int, fillcolor: Any = None) -> Any:
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
        fillcolor=fillcolor,
    )


# from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand  # noqa
def rotate_with_fill(img: Any, magnitude: int) -> Any:
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(
        img.mode
    )


def color(img: Any, magnitude: int) -> Any:
    return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))


def posterize(img: Any, magnitude: int) -> Any:
    return ImageOps.posterize(img, magnitude)


def solarize(img: Any, magnitude: int) -> Any:
    return ImageOps.solarize(img, magnitude)


def contrast(img: Any, magnitude: int) -> Any:
    return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))


def sharpness(img: Any, magnitude: int) -> Any:
    return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))


def brightness(img: Any, magnitude: int) -> Any:
    return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))


def auto_contrast(img: Any, magnitude: int) -> Any:
    return ImageOps.autocontrast(img)


def equalize(img: Any, magnitude: int) -> Any:
    return ImageOps.equalize(img)


def invert(img: Any, magnitude: int) -> Any:
    return ImageOps.invert(img)


def get_image_op_settings(
    image_op: ImageOp, fillcolor: Tuple[int, int, int] = MIDDLE_GRAY
):
    return {
        ImageOp.SHEAR_X: ImageOpSetting(
            np.linspace(0, 0.3, 10), partial(shear_x, fillcolor=fillcolor)
        ),
        ImageOp.SHEAR_Y: ImageOpSetting(
            np.linspace(0, 0.3, 10), partial(shear_y, fillcolor=fillcolor)
        ),
        ImageOp.TRANSLATE_X: ImageOpSetting(
            np.linspace(0, 150 / 331, 10), partial(translate_x, fillcolor=fillcolor)
        ),
        ImageOp.TRANSLATE_Y: ImageOpSetting(
            np.linspace(0, 150 / 331, 10), partial(translate_y, fillcolor=fillcolor)
        ),
        ImageOp.ROTATE: ImageOpSetting(np.linspace(0, 30, 10), rotate_with_fill),
        ImageOp.COLOR: ImageOpSetting(np.linspace(0.0, 0.9, 10), color),
        ImageOp.POSTERIZE: ImageOpSetting(
            np.round(np.linspace(8, 4, 10), 0).astype(np.int), posterize
        ),
        ImageOp.SOLARIZE: ImageOpSetting(np.linspace(256, 0, 10), solarize),
        ImageOp.CONTRAST: ImageOpSetting(np.linspace(0.0, 0.9, 10), contrast),
        ImageOp.SHARPNESS: ImageOpSetting(np.linspace(0.0, 0.9, 10), sharpness),
        ImageOp.BRIGHTNESS: ImageOpSetting(np.linspace(0.0, 0.9, 10), brightness),
        ImageOp.AUTO_CONTRAST: ImageOpSetting([0] * 10, auto_contrast),
        ImageOp.EQUALIZE: ImageOpSetting([0] * 10, equalize),
        ImageOp.INVERT: ImageOpSetting([0] * 10, invert),
    }[image_op]


class SubPolicy:
    def __init__(
        self,
        operation1: ImageOp,
        magnitude_idx1: int,
        p1: float,
        operation2: ImageOp,
        magnitude_idx2: int,
        p2: float,
        fillcolor: Tuple[int, int, int] = MIDDLE_GRAY,
    ) -> None:
        operation1_settings = get_image_op_settings(operation1, fillcolor)
        self.operation1 = operation1_settings.function
        self.magnitude1 = operation1_settings.ranges[magnitude_idx1]
        self.p1 = p1

        operation2_settings = get_image_op_settings(operation2, fillcolor)
        self.operation2 = operation2_settings.function
        self.magnitude2 = operation2_settings.ranges[magnitude_idx2]
        self.p2 = p2

    def __call__(self, img: Any) -> Any:
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


@register_transform("imagenet_autoaugment")
class ImagenetAutoAugment(ClassyTransform):
    """Randomly choose one of the best 24 Sub-policies on ImageNet.

    Example:
    >>> policy = ImageNetPolicy()
    >>> transformed = policy(image)

    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     ImageNetPolicy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor: Tuple[int, int, int] = MIDDLE_GRAY) -> None:
        self.policies = [
            SubPolicy(ImageOp.POSTERIZE, 8, 0.4, ImageOp.ROTATE, 9, 0.6, fillcolor),
            SubPolicy(
                ImageOp.SOLARIZE, 5, 0.6, ImageOp.AUTO_CONTRAST, 5, 0.6, fillcolor
            ),
            SubPolicy(ImageOp.EQUALIZE, 8, 0.8, ImageOp.EQUALIZE, 3, 0.6, fillcolor),
            SubPolicy(ImageOp.POSTERIZE, 7, 0.6, ImageOp.POSTERIZE, 6, 0.6, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 7, 0.4, ImageOp.SOLARIZE, 4, 0.2, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 4, 0.4, ImageOp.ROTATE, 8, 0.8, fillcolor),
            SubPolicy(ImageOp.SOLARIZE, 3, 0.6, ImageOp.EQUALIZE, 7, 0.6, fillcolor),
            SubPolicy(ImageOp.POSTERIZE, 5, 0.8, ImageOp.EQUALIZE, 2, 1.0, fillcolor),
            SubPolicy(ImageOp.ROTATE, 3, 0.2, ImageOp.SOLARIZE, 8, 0.6, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 8, 0.6, ImageOp.POSTERIZE, 6, 0.4, fillcolor),
            SubPolicy(ImageOp.ROTATE, 8, 0.8, ImageOp.COLOR, 0, 0.4, fillcolor),
            SubPolicy(ImageOp.ROTATE, 9, 0.4, ImageOp.EQUALIZE, 2, 0.6, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 7, 0.0, ImageOp.EQUALIZE, 8, 0.8, fillcolor),
            SubPolicy(ImageOp.INVERT, 4, 0.6, ImageOp.EQUALIZE, 8, 1.0, fillcolor),
            SubPolicy(ImageOp.COLOR, 4, 0.6, ImageOp.CONTRAST, 8, 1.0, fillcolor),
            SubPolicy(ImageOp.ROTATE, 8, 0.8, ImageOp.COLOR, 2, 1.0, fillcolor),
            SubPolicy(ImageOp.COLOR, 8, 0.8, ImageOp.SOLARIZE, 7, 0.8, fillcolor),
            SubPolicy(ImageOp.SHARPNESS, 7, 0.4, ImageOp.INVERT, 8, 0.6, fillcolor),
            SubPolicy(ImageOp.SHEAR_X, 5, 0.6, ImageOp.EQUALIZE, 9, 1.0, fillcolor),
            SubPolicy(ImageOp.COLOR, 0, 0.4, ImageOp.EQUALIZE, 3, 0.6, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 7, 0.4, ImageOp.SOLARIZE, 4, 0.2, fillcolor),
            SubPolicy(
                ImageOp.SOLARIZE, 5, 0.6, ImageOp.AUTO_CONTRAST, 5, 0.6, fillcolor
            ),
            SubPolicy(ImageOp.INVERT, 4, 0.6, ImageOp.EQUALIZE, 8, 1.0, fillcolor),
            SubPolicy(ImageOp.COLOR, 4, 0.6, ImageOp.CONTRAST, 8, 1.0, fillcolor),
        ]

    def __call__(self, img: Any) -> Any:
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
