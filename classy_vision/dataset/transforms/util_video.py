#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    CROP_SIZE = 112


@register_transform("video_default_augment")
class VideoDefaultAugmentTransform(ClassyTransform):
    def __init__(
        self,
        crop_size: int = VideoConstants.CROP_SIZE,
        mean: List[float] = VideoConstants.MEAN,
        std: List[float] = VideoConstants.STD,
    ):
        self._transform = transforms.Compose(
            [
                transforms_video.ToTensorVideo(),
                transforms_video.RandomResizedCropVideo(crop_size),
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
