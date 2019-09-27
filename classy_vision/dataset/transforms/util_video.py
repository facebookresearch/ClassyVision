#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional

import torch
import torchvision.transforms as transforms

from . import build_transforms, register_transform
from .util import ImagenetConstants


@register_transform("dummy_audio_transform")
class DummyAudioTransform(object):
    """
    A dummy audio transform. It ignores actual audio data, and returna an empty tensor.
    It is useful when actual audio data is raw waveform and has a varying number of
    waveform samples which makes minibatch assembling impossible
    """

    def __init__(self):
        pass

    def __call__(self, _audio):
        return torch.zeros(0, 1, dtype=torch.float)


class VideoConstants:
    """use the same mean/std from image classification to enable the parameter
    inflation where parameters of 2D conv in image model can be inflated into
    3D conv in video model"""

    MEAN = ImagenetConstants.MEAN
    STD = ImagenetConstants.STD
    CROP_SIZE = 112
    RESIZE = 128


class VideoFieldTransform:
    def __init__(self, transform: Dict[str, Callable], key: str = "input") -> None:
        self.key = key
        self.transform = transform

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        assert isinstance(sample, dict) and self.key in sample, (
            "This transform only supports dicts with key '%s'" % self.key
        )
        assert isinstance(sample[self.key], dict), "sample data is expected be a dict"
        assert isinstance(self.transform, dict), "transform is expected to be a dict"

        for mode, modal_data in sample[self.key].items():
            if mode in self.transform:
                sample[self.key][mode] = self.transform[mode](modal_data)

        return sample


@register_transform("video_default_augment")
def video_default_augment_transform(
    crop_size: int = VideoConstants.CROP_SIZE,
    mean: List[float] = VideoConstants.MEAN,
    std: List[float] = VideoConstants.STD,
) -> Dict[str, Callable]:
    return {
        "video": transforms.Compose(
            [
                transforms.ToTensorVideo(),
                transforms.RandomResizedCropVideo(crop_size),
                transforms.RandomHorizontalFlipVideo(),
                transforms.NormalizeVideo(mean=mean, std=std),
            ]
        ),
        "audio": DummyAudioTransform(),
    }


@register_transform("video_default_no_augment")
def video_default_no_augment_transform(
    resize: int = VideoConstants.RESIZE,
    crop_size: int = VideoConstants.CROP_SIZE,
    mean: List[float] = VideoConstants.MEAN,
    std: List[float] = VideoConstants.STD,
) -> Dict[str, Callable]:
    return {
        "video": transforms.Compose(
            [
                transforms.ToTensorVideo(),
                transforms.CenterCropVideo(crop_size),
                transforms.NormalizeVideo(mean=mean, std=std),
            ]
        ),
        "audio": DummyAudioTransform(),
    }


def build_video_transforms(
    transforms_config: Dict[str, List[Dict[str, Any]]], split: str = "train"
) -> Dict[str, Callable]:
    """
    For each video data modality (.e.g. video, audio), build a transform from the
    list of transform configurations.
    """
    transforms = {}
    for mode, modal_transforms_config in transforms_config.items():
        assert mode in ["video", "audio"], "unknown video data modality %s" % mode
        transforms[mode] = build_transforms(modal_transforms_config)

    video_default_transform = (
        video_default_augment_transform()
        if split == "train"
        else video_default_no_augment_transform()
    )
    for mode, mode_transform in video_default_transform.items():
        # if user does not specify transform for one mode, use the default transform
        if mode not in transforms_config:
            transforms[mode] = mode_transform

    return transforms


def build_video_field_transform_default(
    config: Optional[Dict[str, List[Dict[str, Any]]]],
    split: str = "train",
    key: str = "input",
) -> Callable:
    """
    Returns a VideoFieldTransform which applies a transform on the specified key.

    """
    if config is None:
        transform = (
            video_default_augment_transform()
            if split == "train"
            else video_default_no_augment_transform()
        )
    else:
        transform = build_video_transforms(config, split)
    return VideoFieldTransform(transform, key=key)
