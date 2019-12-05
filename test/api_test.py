#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from typing import Any, Callable, Dict, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from classy_vision.dataset import ClassyDataset, register_dataset
from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
    SampleType,
)
from classy_vision.dataset.transforms import (
    ClassyTransform,
    GenericImageTransform,
    build_transforms,
)
from classy_vision.losses import ClassyLoss, register_loss
from classy_vision.models import ClassyModel, register_model
from classy_vision.optim import SGD
from classy_vision.optim.param_scheduler import ConstantParamScheduler
from classy_vision.tasks import ClassificationTask
from classy_vision.trainer import LocalTrainer
from torchvision import transforms


# WARNING: The goal of this test is to use our public API as advertised in our
# tutorials and make sure everything trains successfully. If you break this
# test, make sure you also update our tutorials.


@register_dataset("my_dataset")
class MyDataset(ClassyDataset):
    def __init__(
        self,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Union[ClassyTransform, Callable]],
        num_samples: int,
        crop_size: int,
        class_ratio: float,
        seed: int,
        split: Optional[str] = None,
    ) -> None:
        dataset = RandomImageBinaryClassDataset(
            crop_size, class_ratio, num_samples, seed, SampleType.TUPLE
        )
        super().__init__(
            dataset, split, batchsize_per_replica, shuffle, transform, num_samples
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MyDataset":
        assert all(key in config for key in ["crop_size", "class_ratio", "seed"])

        split = config.get("split")
        crop_size = config["crop_size"]
        class_ratio = config["class_ratio"]
        seed = config["seed"]
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        transform = build_transforms(transform_config)
        return cls(
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            crop_size,
            class_ratio,
            seed,
            split=split,
        )


@register_loss("my_loss")
class MyLoss(ClassyLoss):
    def forward(self, input, target):
        labels = F.one_hot(target, num_classes=2).float()
        return F.binary_cross_entropy(input, labels)

    @classmethod
    def from_config(cls, config):
        # We don't need anything from the config
        return cls()


@register_model("my_model")
class MyModel(ClassyModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((20, 20)),
            nn.Flatten(1),
            nn.Linear(3 * 20 * 20, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls()


class APITest(unittest.TestCase):
    def testOne(self):
        train_dataset = MyDataset(
            batchsize_per_replica=32,
            shuffle=False,
            transform=GenericImageTransform(
                transform=transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            ),
            num_samples=100,
            crop_size=224,
            class_ratio=0.5,
            seed=0,
        )

        test_dataset = MyDataset(
            batchsize_per_replica=32,
            shuffle=False,
            transform=GenericImageTransform(
                transform=transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            ),
            num_samples=100,
            crop_size=224,
            class_ratio=0.5,
            seed=0,
        )

        model = MyModel()
        loss = MyLoss()

        optimizer = SGD(lr_scheduler=ConstantParamScheduler(0.01))

        task = (
            ClassificationTask()
            .set_model(model)
            .set_dataset(train_dataset, "train")
            .set_dataset(test_dataset, "test")
            .set_loss(loss)
            .set_optimizer(optimizer)
            .set_num_epochs(1)
        )

        trainer = LocalTrainer()
        trainer.train(task)
