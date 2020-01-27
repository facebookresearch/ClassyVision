#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torchvision.transforms as transforms

from . import ClassyTransform, build_transforms, register_transform


class ImagenetConstants:
    """Constant variables related to the image classification.

    MEAN: often used to be subtracted from image RGB value. Computed on ImageNet.
    STD: often used to divide the image RGB value after mean centering. Computed
        on ImageNet.
    CROP_SIZE: the size of image cropping which is often the input to deep network.
    RESIZE: the size of rescaled image.

    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    CROP_SIZE = 224
    RESIZE = 256


@register_transform("apply_transform_to_key")
class ApplyTransformToKey:
    """Serializable class that applies a transform to a key specified field in samples.
    """

    def __init__(self, transform: Callable, key: Union[int, str] = "input") -> None:
        """The constructor method of ApplyTransformToKey class.

        Args:
            transform: a callable function that takes sample data of type dict as input
            key: the key in sample whose corresponding value will undergo
                the transform

        """
        self.key: Union[int, str] = key
        self.transform: Callable = transform

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        transform = build_transforms(config["transforms"])

        return cls(transform=transform, key=config["key"])

    def __call__(
        self, sample: Union[Tuple[Any], Dict[str, Any]]
    ) -> Union[Tuple[Any], Dict[str, Any]]:
        """Updates sample by applying a transform to the value at the specified key.

        Args:
            sample: input sample which will be transformed

        """
        if sample is None:
            return sample

        # Asserts + deal with tuple immutability
        convert_to_tuple = False
        if isinstance(sample, dict):
            assert (
                self.key in sample
            ), "This transform only supports dicts with key '{}'".format(self.key)
        elif isinstance(sample, (tuple, list)):
            assert self.key < len(
                sample
            ), "This transform only supports tuples / lists with key less "
            "than {length}, key provided {key}".format(length=len(sample), key=self.key)
            # Convert to list for transformation
            if isinstance(sample, tuple):
                convert_to_tuple = True
            sample = list(sample)

        sample[self.key] = self.transform(sample[self.key])
        if convert_to_tuple:
            sample = tuple(sample)

        return sample


@register_transform("imagenet_augment")
class ImagenetAugmentTransform(ClassyTransform):
    """The default image transform with data augmentation.

    It is often useful for training models on Imagenet. It sequentially resizes
    the image into a random scale, takes a random spatial cropping, randomly flips
    the image horizontally, transforms PIL image data into a torch.Tensor and
    normalizes the pixel values by mean subtraction and standard deviation division.
    """

    def __init__(
        self,
        crop_size: int = ImagenetConstants.CROP_SIZE,
        mean: List[float] = ImagenetConstants.MEAN,
        std: List[float] = ImagenetConstants.STD,
    ):
        """The constructor method of ImagenetAugmentTransform class.

        Args:
            crop_size: expected output size per dimension after random cropping
            mean: a 3-tuple denoting the pixel RGB mean
            std: a 3-tuple denoting the pixel RGB standard deviation

        """
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        """Callable function which applies the tranform to the input image.

        Args:
            image: input image that will undergo the transform

        """
        return self.transform(img)


@register_transform("imagenet_no_augment")
class ImagenetNoAugmentTransform(ClassyTransform):
    """The default image transform without data augmentation.

    It is often useful for testing models on Imagenet. It sequentially resizes
    the image, takes a central  cropping, transforms PIL image data into a
    torch.Tensor and normalizes the pixel values by mean subtraction and standard
    deviation division.

    """

    def __init__(
        self,
        resize: int = ImagenetConstants.RESIZE,
        crop_size: int = ImagenetConstants.CROP_SIZE,
        mean: List[float] = ImagenetConstants.MEAN,
        std: List[float] = ImagenetConstants.STD,
    ):
        """The constructor method of ImagenetNoAugmentTransform class.

        Args:
            resize: expected image size per dimension after resizing
            crop_size: expected size for a dimension of central cropping
            mean: a 3-tuple denoting the pixel RGB mean
            std: a 3-tuple denoting the pixel RGB standard deviation

        """
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        """Callable function which applies the tranform to the input image.

        Args:
            image: input image that will undergo the transform

        """
        return self.transform(img)


@register_transform("generic_image_transform")
class GenericImageTransform(ClassyTransform):
    """Default transform for images used in the classification task

    This transform does several things. First, it expects a tuple or
    list input (torchvision datasets supply tuples / lists). Second,
    it applies a user-provided image transforms to the first entry in
    the tuple (again, matching the torchvision tuple format). Third,
    it transforms the tuple to a dict sample with entries "input" and
    "target".

    The defaults are for the standard imagenet augmentations

    This is just a convenience wrapper to cover the common
    use-case. You can get the same behavior by composing `torchvision
    transforms <https://pytorch.org/docs/stable/torchvision/transforms.html>`_
    + :class:`ApplyTransformToKey` + :class:`TupleToMapTransform`.

    """

    def __init__(
        self, transform: Optional[Callable] = None, split: Optional[str] = None
    ):
        """Constructor for GenericImageTransfrom
        Only one of the two arguments (*transform*, *split*) should be specified.
        Args:
            transform: A callable or ClassyTransform to be applied to the image only
            split: 'train' or 'test'
        """
        assert (
            split is None or transform is None
        ), "If split is not None then transform must be None"
        assert split in [None, "train", "test"], (
            "If specified, split should be either 'train' or 'test', "
            "instead got {}".format(split)
        )

        self._transform = transform
        if split is not None:
            self._transform = (
                ImagenetAugmentTransform()
                if split == "train"
                else ImagenetNoAugmentTransform()
            )

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        transform = None
        if "transforms" in config:
            transform = build_transforms(config["transforms"])
        split = config.get("split")
        return cls(transform, split)

    def __call__(self, sample: Tuple[Any]):
        """Applied transform to sample

        Args:
            sample: A tuple with length >= 2. The first entry should
                be the image data, the second entry should be the
                target data.
        """
        image = sample[0]
        transformed_image = (
            self._transform(image) if self._transform is not None else image
        )
        new_sample = {"input": transformed_image, "target": sample[1]}
        # Any additional metadata is just appended under index of tuple
        if len(sample) > 2:
            for i in range(2, len(sample)):
                new_sample[str(i)] = sample[i]

        return new_sample


@register_transform("tuple_to_map")
class TupleToMapTransform(ClassyTransform):
    """A transform which maps image data from tuple to dict.

    This transform has a list of keys (key1, key2, ...),
    takes a sample of the form (data1, data2, ...) and
    returns a sample of the form {key1: data1, key2: data2, ...}

    It is useful for mapping output from datasets like the `PyTorch
    ImageFolder <https://github.com/pytorch/vision/blob/master/torchvision/
    datasets/folder.py#L177>`_ dataset (tuple) to dict with named data fields.

    If sample is already a dict with the required keys, pass sample through.

    """

    def __init__(self, list_of_map_keys: List[str]):
        """The constructor method of TupleToMapTransform class.

        Args:
            list_of_map_keys: a list of dict keys that in order will be mapped
                to items in the input data sample list

        """
        self._map_keys = list_of_map_keys

    def __call__(self, sample):
        """Transform sample from type tuple to type dict.

        Args:
            sample: input sample which will be transformed

        """
        # If already a dict/map with appropriate keys, exit early
        if isinstance(sample, dict):
            for key in self._map_keys:
                assert (
                    key in sample
                ), "Sample {sample} must be a tuple or a dict with keys {keys}".format(
                    sample=str(sample), keys=str(self._map_keys)
                )
            return sample

        assert len(sample) == len(self._map_keys), (
            "Provided sample tuple must have same number of keys "
            "as provided to transform"
        )
        output_sample = {}
        for idx, s in enumerate(sample):
            output_sample[self._map_keys[idx]] = s

        return output_sample


DEFAULT_KEY_MAP = TupleToMapTransform(["input", "target"])


def build_field_transform_default_imagenet(
    config: Optional[List[Dict[str, Any]]],
    default_transform: Optional[Callable] = None,
    split: Optional[bool] = None,
    key: Union[int, str] = "input",
    key_map_transform: Optional[Callable] = DEFAULT_KEY_MAP,
) -> Callable:
    """Returns a ApplyTransformToKey which applies a transform on the specified key.

    The transform is built from the config, if it is not None.

    Otherwise, uses one of the two mutually exclusive args: If
    default_transform is not None, it is used.  If split is not None,
    imagenet transforms are used, using augmentation for "train", no
    augmentation otherwise.

    This function also provides an additional
    function for mapping from tuples (or other keys) to a desired set
    of keys

    Args:
        config: field transform config
        default_transform: used if config is None
        split: split for dataset, e.g. "train" or "test"
        key: Key to apply transform to
        key_map_transform: Used to produce desired map / keys
            (e.g. for torchvision datasets, default samples is a
            tuple so this argument can be used to map
            (input, target) -> {"input": input, "target": target})

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

    transform = ApplyTransformToKey(transform, key=key)
    if key_map_transform is None:
        return transform

    return transforms.Compose([key_map_transform, transform])


def default_unnormalize(img):
    """Default unnormalization transform which undo the "transforms.Normalize".

        Specially, it cancels out mean subtraction and standard deviation division.

    Args:
        img (torch.Tensor): image data to which the transform will be applied

    """
    # TODO T39752655: Allow this to be configurable
    img = img.clone()
    for channel, std, mean in zip(img, ImagenetConstants.STD, ImagenetConstants.MEAN):
        channel.mul_(std).add_(mean)
    return img
