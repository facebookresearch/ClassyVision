#!/usr/bin/env python3

import unittest

import torch
import torchvision.transforms as transforms
from classy_vision.dataset.transforms import build_transforms, register_transform
from classy_vision.dataset.transforms.util import imagenet_no_augment_transform


@register_transform("resize")
def resize(size: int):
    return transforms.Resize(size=size)


@register_transform("center_crop")
def center_crop(size: int):
    return transforms.CenterCrop(size=size)


class DatasetTransformsTest(unittest.TestCase):
    def get_test_image(self):
        return transforms.ToPILImage()(torch.randn((3, 224, 224)))

    def test_transforms(self):
        input = self.get_test_image()

        # reference transform which we will use to validate the built transforms
        reference_transform = imagenet_no_augment_transform()
        reference_output = reference_transform(input)

        # test a registered transform
        config = [{"name": "imagenet_no_augment"}]
        transform = build_transforms(config)
        output = transform(input)
        self.assertTrue(torch.allclose(output, reference_output))

        # test a transform built using torchvision transforms
        config = [
            {"name": "Resize", "size": 256},
            {"name": "CenterCrop", "size": 224},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ]
        transform = build_transforms(config)
        output = transform(input)
        self.assertTrue(torch.allclose(output, reference_output))

        # test a combination of registered and torchvision transforms
        config = [
            {"name": "resize", "size": 256},
            {"name": "center_crop", "size": 224},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ]
        transform = build_transforms(config)
        output = transform(input)
        self.assertTrue(torch.allclose(output, reference_output))
