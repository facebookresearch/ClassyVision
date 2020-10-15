#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from . import ClassyTransform, build_transforms, register_transform
from .util import ApplyTransformToKey, ImagenetConstants, TupleToMapTransform


class VideoConstants:
    """Constant variables related to the video classification.

    Use the same mean/std from image classification to enable the parameter
    inflation where parameters of 2D conv in image model can be inflated into
    3D conv in video model.

    MEAN: often used to be subtracted from pixel RGB value.
    STD: often used to divide the pixel RGB value after mean centering.
    SIZE_RANGE: a (min_size, max_size) tuple which denotes the range of
        size of the rescaled video clip.
    CROP_SIZE: the size of spatial cropping in the video clip.
    """

    MEAN = ImagenetConstants.MEAN  #
    STD = ImagenetConstants.STD
    SIZE_RANGE = (128, 160)
    CROP_SIZE = 112


def _get_rescaled_size(scale, h, w):
    if h < w:
        new_h = scale
        new_w = int(scale * w / h)
    else:
        new_w = scale
        new_h = int(scale * h / w)
    return new_h, new_w


@register_transform("video_clip_random_resize_crop")
class VideoClipRandomResizeCrop(ClassyTransform):
    """A video clip transform that is often useful for trainig data.

    Given a size range, randomly choose a size. Rescale the clip so that
    its short edge equals to the chosen size. Then randomly crop the video
    clip with the specified size.
    Such training data augmentation is used in VGG net
    (https://arxiv.org/abs/1409.1556).
    Also see reference implementation `Kinetics.spatial_sampling` in SlowFast
        codebase.
    """

    def __init__(
        self,
        crop_size: Union[int, List[int]],
        size_range: List[int],
        interpolation_mode: str = "bilinear",
    ):
        """The constructor method of VideoClipRandomResizeCrop class.

        Args:
            crop_size: int or 2-tuple as the expected output crop_size (height, width)
            size_range: the min- and max size
            interpolation_mode: Default: "bilinear"

        """
        if isinstance(crop_size, tuple):
            assert len(crop_size) == 2, "crop_size should be tuple (height, width)"
            self.crop_size = crop_size
        else:
            self.crop_size = (crop_size, crop_size)

        self.interpolation_mode = interpolation_mode
        self.size_range = size_range

    def __call__(self, clip):
        """Callable function which applies the tranform to the input clip.

        Args:
            clip (torch.Tensor): input clip tensor

        """
        # clip size: C x T x H x W
        rand_size = random.randint(self.size_range[0], self.size_range[1])
        new_h, new_w = _get_rescaled_size(rand_size, clip.size()[2], clip.size()[3])
        clip = torch.nn.functional.interpolate(
            clip, size=(new_h, new_w), mode=self.interpolation_mode
        )
        assert (
            self.crop_size[0] <= new_h and self.crop_size[1] <= new_w
        ), "crop size can not be larger than video frame size"

        i = random.randint(0, new_h - self.crop_size[0])
        j = random.randint(0, new_w - self.crop_size[1])
        clip = clip[:, :, i : i + self.crop_size[0], j : j + self.crop_size[1]]
        return clip


@register_transform("video_clip_resize")
class VideoClipResize(ClassyTransform):
    """A video clip transform that is often useful for testing data.

    Given an input size, rescale the clip so that its short edge equals to
    the input size while aspect ratio is preserved.
    """

    def __init__(self, size: int, interpolation_mode: str = "bilinear"):
        """The constructor method of VideoClipResize class.

        Args:
            size: input size
            interpolation_mode: Default: "bilinear". See valid values in
                (https://pytorch.org/docs/stable/nn.functional.html#torch.nn.
                functional.interpolate)

        """
        self.interpolation_mode = interpolation_mode
        self.size = size

    def __call__(self, clip):
        """Callable function which applies the tranform to the input clip.

        Args:
            clip (torch.Tensor): input clip tensor

        """
        # clip size: C x T x H x W
        if not min(clip.size()[2], clip.size()[3]) == self.size:
            new_h, new_w = _get_rescaled_size(self.size, clip.size()[2], clip.size()[3])
            clip = torch.nn.functional.interpolate(
                clip, size=(new_h, new_w), mode=self.interpolation_mode
            )
        return clip


@register_transform("video_default_augment")
class VideoDefaultAugmentTransform(ClassyTransform):
    """This is the default video transform with data augmentation which is useful for
    training.

    It sequentially prepares a torch.Tensor of video data, randomly
    resizes the video clip, takes a random spatial cropping, randomly flips the
    video clip horizontally, and normalizes the pixel values by mean subtraction
    and standard deviation division.

    """

    def __init__(
        self,
        crop_size: Union[int, List[int]] = VideoConstants.CROP_SIZE,
        size_range: List[int] = VideoConstants.SIZE_RANGE,
        mean: List[float] = VideoConstants.MEAN,
        std: List[float] = VideoConstants.STD,
    ):
        """The constructor method of VideoDefaultAugmentTransform class.

        Args:
            crop_size: expected output crop_size (height, width)
            size_range : a 2-tuple denoting the min- and max size
            mean: a 3-tuple denoting the pixel RGB mean
            std: a 3-tuple denoting the pixel RGB standard deviation

        """

        self._transform = transforms.Compose(
            [
                transforms_video.ToTensorVideo(),
                # TODO(zyan3): migrate VideoClipRandomResizeCrop to TorchVision
                VideoClipRandomResizeCrop(crop_size, size_range),
                transforms_video.RandomHorizontalFlipVideo(),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
        )

    def __call__(self, video):
        """Apply the default transform with data augmentation to video.

        Args:
            video: input video that will undergo the transform

        """
        return self._transform(video)


@register_transform("video_default_no_augment")
class VideoDefaultNoAugmentTransform(ClassyTransform):
    """This is the default video transform without data augmentation which is useful
    for testing.

    It sequentially prepares a torch.Tensor of video data, resize the
    video clip to have the specified short edge, and normalize the pixel values
    by mean subtraction and standard deviation division.

    """

    def __init__(
        self,
        size: int = VideoConstants.SIZE_RANGE[0],
        mean: List[float] = VideoConstants.MEAN,
        std: List[float] = VideoConstants.STD,
    ):
        """The constructor method of VideoDefaultNoAugmentTransform class.

        Args:
            size: the short edge of rescaled video clip
            mean: a 3-tuple denoting the pixel RGB mean
            std: a 3-tuple denoting the pixel RGB standard deviation

        """
        self._transform = transforms.Compose(
            # At testing stage, central cropping is not used because we
            # conduct fully convolutional-style testing
            [
                transforms_video.ToTensorVideo(),
                # TODO(zyan3): migrate VideoClipResize to TorchVision
                VideoClipResize(size),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
        )

    def __call__(self, video):
        """Apply the default transform without data augmentation to video.

        Args:
            video: input video that will undergo the transform

        """
        return self._transform(video)


@register_transform("dummy_audio_transform")
class DummyAudioTransform(ClassyTransform):
    """This is a dummy audio transform.

    It ignores actual audio data, and returns an empty tensor. It is useful when
    actual audio data is raw waveform and has a varying number of waveform samples
    which makes minibatch assembling impossible

    """

    def __init__(self):
        """The constructor method of DummyAudioTransform class."""

        pass

    def __call__(self, _audio):
        """Callable function which applies the tranform to the input audio data.

        Args:
            audio: input audio data that will undergo the dummy transform

        """
        return torch.zeros(0, 1, dtype=torch.float)


# Maps (video, audio, target) tuple to {'input': (video, audio), 'target': target}
DEFAULT_KEY_MAP = TupleToMapTransform(["input", "input", "target"])


def build_video_field_transform_default(
    config: Optional[Dict[str, List[Dict[str, Any]]]],
    split: str = "train",
    key: str = "input",
    key_map_transform: Optional[Callable] = DEFAULT_KEY_MAP,
) -> Callable:
    """Returns transform that first maps sample to video keys, then
    returns a transform on the specified key in dict.

    Converts tuple (list, etc) sample to dict with input / target keys.
    For a dict sample, verifies that dict has input / target keys.
    For all other samples throws.

    Args:
        config: If provided, it is a dict where key is the data modality, and
            value is a dict specifying the transform config
        split: the split of the data to which the transform will be applied
        key: the key in data sample of type dict whose corresponding value will
            undergo the transform
    """
    if config is None and split is None:
        raise ValueError("No transform config provided with no defaults")

    transforms_for_type = {
        "video": VideoDefaultAugmentTransform()
        if split == "train"
        else VideoDefaultNoAugmentTransform(),
        "audio": DummyAudioTransform(),
    }

    if config is not None:
        transforms_for_type.update(
            {
                mode: build_transforms(modal_config)
                for mode, modal_config in config.items()
            }
        )

    transform = transforms.Compose(
        [
            ApplyTransformToKey(default_transform, key=mode)
            for mode, default_transform in transforms_for_type.items()
        ]
    )

    transform = ApplyTransformToKey(
        transforms.Compose([TupleToMapTransform(["video", "audio"]), transform]),
        key=key,
    )
    if key_map_transform is None:
        return transform

    return transforms.Compose([key_map_transform, transform])
