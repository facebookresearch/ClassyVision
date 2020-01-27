#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
import unittest

import numpy
import torch
import torchvision.transforms as transforms
from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
    SampleType,
)
from classy_vision.dataset.transforms import build_transforms
from classy_vision.dataset.transforms.util import (
    GenericImageTransform,
    ImagenetAugmentTransform,
    ImagenetNoAugmentTransform,
    build_field_transform_default_imagenet,
)


def _apply_transform_to_key_and_copy(sample, transform, key, seed=0):
    """
    This helper function takes a sample, makes a copy, applies the
    provided transform to the appropriate key in the copied sample and
    returns the copy. It's solely to help make sure the copying /
    random seed happens correctly throughout the file.

    It is useful for constructing the expected sample field in the
    transform checks.
    """
    expected_sample = copy.deepcopy(sample)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    is_tuple = False
    if isinstance(expected_sample, tuple):
        expected_sample = list(expected_sample)
        is_tuple = True
    expected_sample[key] = transform(expected_sample[key])
    return tuple(expected_sample) if is_tuple else expected_sample


class DatasetTransformsUtilTest(unittest.TestCase):
    def get_test_image_dataset(self, sample_type):
        return RandomImageBinaryClassDataset(
            crop_size=224,
            class_ratio=0.5,
            num_samples=100,
            seed=0,
            sample_type=sample_type,
        )

    def transform_checks(self, sample, transform, expected_sample, seed=0):
        """
        This helper function applies the transform to the sample
        and verifies that the output is the expected_sample. The
        sole purpose is to make sure copying / random seed / checking
        all of the fields in the sample happens correctly.
        """
        transformed_sample = copy.deepcopy(sample)
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        transformed_sample = transform(transformed_sample)

        if isinstance(expected_sample, (tuple, list)):
            for transformed, exp in zip(transformed_sample, expected_sample):
                if torch.is_tensor(exp):
                    self.assertTrue(torch.allclose(transformed, exp))

        if isinstance(expected_sample, dict):
            for key, exp_val in expected_sample.items():
                self.assertTrue(key in transformed_sample)
                if torch.is_tensor(exp_val):
                    self.assertTrue(torch.allclose(transformed_sample[key], exp_val))
                elif isinstance(exp_val, float):
                    self.assertAlmostEqual(transformed_sample[key], exp_val)
                else:
                    self.assertEqual(transformed_sample[key], exp_val)

    def test_build_dict_field_transform_default_imagenet(self):
        dataset = self.get_test_image_dataset(SampleType.DICT)

        # should apply the transform in the config
        config = [{"name": "ToTensor"}]
        default_transform = transforms.Compose(
            [transforms.CenterCrop(100), transforms.ToTensor()]
        )
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform
        )
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            sample, transforms.ToTensor(), "input"
        )
        self.transform_checks(sample, transform, expected_sample)

        # should apply default_transform
        config = None
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform
        )
        expected_sample = _apply_transform_to_key_and_copy(
            sample, default_transform, "input"
        )
        self.transform_checks(sample, transform, expected_sample)

        # should apply the transform for a test split
        transform = build_field_transform_default_imagenet(config, split="test")
        expected_sample = _apply_transform_to_key_and_copy(
            sample, ImagenetNoAugmentTransform(), "input"
        )
        self.transform_checks(sample, transform, expected_sample)

    def test_build_tuple_field_transform_default_imagenet(self):
        dataset = self.get_test_image_dataset(SampleType.TUPLE)

        # should apply the transform in the config
        config = [{"name": "ToTensor"}]
        default_transform = transforms.Compose(
            [transforms.CenterCrop(100), transforms.ToTensor()]
        )
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform, key=0, key_map_transform=None
        )
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            sample, transforms.ToTensor(), 0
        )
        self.transform_checks(sample, transform, expected_sample)

        # should apply default_transform
        config = None
        transform = build_field_transform_default_imagenet(
            config, default_transform=default_transform, key=0, key_map_transform=None
        )
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(sample, default_transform, 0)
        self.transform_checks(sample, transform, expected_sample)

        # should apply the transform for a test split
        transform = build_field_transform_default_imagenet(
            config, split="test", key=0, key_map_transform=None
        )
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            sample, ImagenetNoAugmentTransform(), 0
        )
        self.transform_checks(sample, transform, expected_sample)

    def test_apply_transform_to_key_from_config(self):
        dataset = self.get_test_image_dataset(SampleType.DICT)

        config = [
            {
                "name": "apply_transform_to_key",
                "transforms": [{"name": "ToTensor"}],
                "key": "input",
            }
        ]
        transform = build_transforms(config)
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            sample, transforms.ToTensor(), "input"
        )
        self.transform_checks(sample, transform, expected_sample)

    def test_generic_image_transform(self):
        dataset = self.get_test_image_dataset(SampleType.TUPLE)

        # Check constructor asserts
        with self.assertRaises(AssertionError):
            transform = GenericImageTransform(
                split="train", transform=transforms.ToTensor()
            )
            transform = GenericImageTransform(split="valid", transform=None)

        # Check class constructor
        transform = GenericImageTransform(transform=None)
        PIL_sample = dataset[0]
        tensor_sample = (transforms.ToTensor()(PIL_sample[0]), PIL_sample[1])
        expected_sample = {
            "input": copy.deepcopy(tensor_sample[0]),
            "target": copy.deepcopy(tensor_sample[1]),
        }
        self.transform_checks(tensor_sample, transform, expected_sample)

        transform = GenericImageTransform(transform=transforms.ToTensor())
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            {"input": sample[0], "target": sample[1]}, transforms.ToTensor(), "input"
        )
        self.transform_checks(sample, transform, expected_sample)

        transform = GenericImageTransform(split="train")
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            {"input": sample[0], "target": sample[1]},
            ImagenetAugmentTransform(),
            "input",
        )
        self.transform_checks(sample, transform, expected_sample)

        transform = GenericImageTransform(split="test")
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            {"input": sample[0], "target": sample[1]},
            ImagenetNoAugmentTransform(),
            "input",
        )
        self.transform_checks(sample, transform, expected_sample)

        # Check from_config constructor / registry
        config = [
            {"name": "generic_image_transform", "transforms": [{"name": "ToTensor"}]}
        ]
        transform = build_transforms(config)
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            {"input": sample[0], "target": sample[1]}, transforms.ToTensor(), "input"
        )
        self.transform_checks(sample, transform, expected_sample)

        # Check with Imagenet defaults
        config = [{"name": "generic_image_transform", "split": "train"}]
        transform = build_transforms(config)
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            {"input": sample[0], "target": sample[1]},
            ImagenetAugmentTransform(),
            "input",
        )
        self.transform_checks(sample, transform, expected_sample)

        config = [{"name": "generic_image_transform", "split": "test"}]
        transform = build_transforms(config)
        sample = dataset[0]
        expected_sample = _apply_transform_to_key_and_copy(
            {"input": sample[0], "target": sample[1]},
            ImagenetNoAugmentTransform(),
            "input",
        )
        self.transform_checks(sample, transform, expected_sample)
