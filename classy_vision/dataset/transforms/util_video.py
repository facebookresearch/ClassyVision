#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Callable, Dict, List, Optional

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from . import ClassyTransform, build_transforms, register_transform
from .util import FieldTransform, ImagenetConstants


class VideoConstants:
    """use the same mean/std from image classification to enable the parameter
    inflation where parameters of 2D conv in image model can be inflated into
    3D conv in video model"""

    MEAN = ImagenetConstants.MEAN
    STD = ImagenetConstants.STD
    SCALE = (128, 160)
    CROP_SIZE = 112


@register_transform("video_random_scale_crop")
class VideoRandomScaleCrop(ClassyTransform):
    """Given a scale range, randomly choose a scale. Rescale the clip so that
    its short edge equals to the chosen scale. Then randomly crop the video
    clip with the specified size.
    Such training data augmengation is used in VGG net
    (https://arxiv.org/abs/1409.1556).
    Also see reference implementation `Kinetics.spatial_sampling` in SlowFast
        codebase.

    Args:
        size (int or tuple): expected output size (height, width)
        scale (2-tuple): the min- and max scale
        interpolation_mode: Default: "bilinear"
    """

    def __init__(self, size, scale, interpolation_mode="bilinear"):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale

    def __call__(self, clip):
        # clip size: C x T x H x W
        h = clip.size()[2]
        w = clip.size()[3]
        rand_scale = random.randint(self.scale[0], self.scale[1])
        if h < w:
            new_h = rand_scale
            new_w = int(rand_scale * w / h)
        else:
            new_w = rand_scale
            new_h = int(rand_scale * h / w)
        clip = torch.nn.functional.interpolate(
            clip, size=(new_h, new_w), mode=self.interpolation_mode
        )
        assert (
            self.size[0] <= new_h and self.size[1] <= new_w
        ), "crop size can not be larger than video frame size"

        i = random.randint(0, new_h - self.size[0])
        j = random.randint(0, new_w - self.size[1])
        clip = clip[:, :, i : i + self.size[0], j : j + self.size[1]]
        return clip


@register_transform("video_default_augment")
class VideoDefaultAugmentTransform(ClassyTransform):
    def __init__(
        self,
        crop_size: int = VideoConstants.CROP_SIZE,
        scale: List[int] = VideoConstants.SCALE,
        mean: List[float] = VideoConstants.MEAN,
        std: List[float] = VideoConstants.STD,
    ):
        self._transform = transforms.Compose(
            [
                transforms_video.ToTensorVideo(),
                # TODO(zyan3): migrate VideoRandomScaleCrop to TorchVision
                VideoRandomScaleCrop(crop_size, scale),
                transforms_video.RandomHorizontalFlipVideo(),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
        )

    def __call__(self, video):
        return self._transform(video)


@register_transform("video_default_no_augment")
class VideoDefaultNoAugmentTransform(ClassyTransform):
    def __init__(
        self,
        mean: List[float] = VideoConstants.MEAN,
        std: List[float] = VideoConstants.STD,
    ):
        self._transform = transforms.Compose(
            # At testing stage, central cropping is not used because we
            # conduct fully convolutional-style testing
            [
                transforms_video.ToTensorVideo(),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
        )

    def __call__(self, video):
        return self._transform(video)


@register_transform("dummy_audio_transform")
class DummyAudioTransform(ClassyTransform):
    """
    A dummy audio transform. It ignores actual audio data, and returns an empty tensor.
    It is useful when actual audio data is raw waveform and has a varying number of
    waveform samples which makes minibatch assembling impossible
    """

    def __init__(self):
        pass

    def __call__(self, _audio):
        return torch.zeros(0, 1, dtype=torch.float)


class ClassyVideoGenericTransform(object):
    def __init__(
        self,
        config: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        split: str = "train",
    ):
        self.transforms = {
            "video": VideoDefaultAugmentTransform()
            if split == "train"
            else VideoDefaultNoAugmentTransform(),
            "audio": DummyAudioTransform(),
        }
        if config is not None:
            for mode, modal_config in config.items():
                assert mode in ["video", "audio"], (
                    "unknown video data modality %s" % mode
                )
                self.transforms[mode] = build_transforms(modal_config)

    def __call__(self, video):
        assert isinstance(video, dict), "video data is expected be a dict"
        for mode, modal_data in video.items():
            if mode in self.transforms:
                video[mode] = self.transforms[mode](modal_data)
        return video


def build_video_field_transform_default(
    config: Optional[Dict[str, List[Dict[str, Any]]]],
    split: str = "train",
    key: str = "input",
) -> Callable:
    """
    Returns a FieldTransform which applies a transform on the specified key.

    """
    transform = ClassyVideoGenericTransform(config, split)
    return FieldTransform(transform, key=key)
