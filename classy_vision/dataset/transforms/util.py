#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict, List, Optional, Union

import torchvision.transforms as transforms

from . import ClassyTransform, build_transforms, register_transform


class ImagenetConstants:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    CROP_SIZE = 224
    RESIZE = 256


class FieldTransform:
    """
    Serializable class that applies a transform on specific field in samples.
    """

    def __init__(self, transform: Callable, key: str = "input") -> None:
        self.key: str = key
        self.transform: Callable = transform

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Updates sample by applying a transform and to the appropriate key."""
        if sample is None:
            return sample

        assert isinstance(sample, dict) and self.key in sample, (
            "This transform only supports dicts with key '%s'" % self.key
        )
        sample[self.key] = self.transform(sample[self.key])
        return sample


@register_transform("imagenet_augment")
class ImagenetAugmentTransform(ClassyTransform):
    def __init__(
        self,
        crop_size: int = ImagenetConstants.CROP_SIZE,
        mean: List[float] = ImagenetConstants.MEAN,
        std: List[float] = ImagenetConstants.STD,
    ):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


@register_transform("imagenet_no_augment")
class ImagenetNoAugmentTransform(ClassyTransform):
    def __init__(
        self,
        resize: int = ImagenetConstants.RESIZE,
        crop_size: int = ImagenetConstants.CROP_SIZE,
        mean: List[float] = ImagenetConstants.MEAN,
        std: List[float] = ImagenetConstants.STD,
    ):
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


@register_transform("tuple_to_map")
class TupleToMapTransform(ClassyTransform):
    """
    This transform takes a sample of the form (data1, data2, ...) and
    returns a sample of the form {key1: data1, key2: data2, ...}

    It is useful for mapping output from datasets like the PyTorch
    ImageFolder dataset (tuple) to dict with named data fields.
    """

    def __init__(self, list_of_map_keys: List[str]):
        self._map_keys = list_of_map_keys

    def __call__(self, sample):
        assert len(sample) == len(self._map_keys), (
            "Provided sample tuple must have same number of keys "
            "as provided to transform"
        )
        output_sample = {}
        for idx, s in enumerate(sample):
            output_sample[self._map_keys[idx]] = s

        return output_sample


def build_field_transform_default_imagenet(
    config: Optional[List[Dict[str, Any]]],
    default_transform: Optional[Callable] = None,
    split: Optional[bool] = None,
    key: str = "input",
) -> Callable:
    """
    Returns a FieldTransform which applies a transform on the specified key.

    The transform is built from the config, if it is not None.
    Otherwise, uses one of the two mutually exclusive args:
        If default_transform is not None, it is used.
        If split is not None, imagenet transforms are used, using augmentation
            for "train", no augmentation otherwise.
    """
    assert (
        default_transform is None or split is None
    ), "Can only specify one of default_transform and split"
    if config is None:
        if default_transform is not None:
            transform = default_transform
        elif split is not None:
            transform = (
                ImagenetAugmentTransform()
                if split == "train"
                else ImagenetNoAugmentTransform()
            )
        else:
            raise ValueError("No transform config provided with no defaults")
    else:
        transform = build_transforms(config)
    return FieldTransform(transform, key=key)


def default_unnormalize(img):
    # TODO T39752655: Allow this to be configurable
    img = img.clone()
    for channel, std, mean in zip(img, ImagenetConstants.STD, ImagenetConstants.MEAN):
        channel.mul_(std).add_(mean)
    return img
