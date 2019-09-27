#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional

import torchvision.transforms as transforms

from . import build_transforms, register_transform
from .util import FieldTransform, ImagenetConstants


class VideoConstants:
    """use the same mean/std from image classification to enable the parameter
    inflation where parameters of 2D conv in image model can be inflated into
    3D conv in video model"""

    MEAN = ImagenetConstants.MEAN
    STD = ImagenetConstants.STD
    CROP_SIZE = 112
    RESIZE = 128


@register_transform("video_default_augment")
def video_augment_transform(
    crop_size: int = VideoConstants.CROP_SIZE,
    mean: List[float] = VideoConstants.MEAN,
    std: List[float] = VideoConstants.STD,
) -> Callable:
    return transforms.Compose(
        [
            transforms.ToTensorVideo(),
            transforms.RandomResizedCropVideo(crop_size),
            transforms.RandomHorizontalFlipVideo(),
            transforms.NormalizeVideo(mean=mean, std=std),
        ]
    )


@register_transform("video_default_no_augment")
def video_no_augment_transform(
    resize: int = VideoConstants.RESIZE,
    crop_size: int = VideoConstants.CROP_SIZE,
    mean: List[float] = VideoConstants.MEAN,
    std: List[float] = VideoConstants.STD,
) -> Callable:
    return transforms.Compose(
        [
            transforms.ToTensorVideo(),
            transforms.CenterCropVideo(crop_size),
            transforms.NormalizeVideo(mean=mean, std=std),
        ]
    )


def build_field_transform_default_video(
    config: Optional[List[Dict[str, Any]]], split: str = "train", key: str = "input"
) -> Callable:
    """
    Returns a FieldTransform which applies a transform on the specified key.

    """
    if config is None:
        transform = (
            video_augment_transform()
            if split == "train"
            else video_no_augment_transform()
        )
    else:
        transform = build_transforms(config)
    return FieldTransform(transform, key=key)
