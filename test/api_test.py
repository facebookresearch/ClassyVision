# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch.nn as nn
import torch.nn.functional as F
from classy_vision.dataset import register_dataset
from classy_vision.dataset.classy_synthetic_image import SyntheticImageDataset
from classy_vision.dataset.transforms import ApplyTransformToKey
from classy_vision.losses import ClassyLoss, register_loss
from classy_vision.models import ClassyModel, register_model
from classy_vision.optim import SGD
from classy_vision.optim.param_scheduler import ConstantParamScheduler
from classy_vision.tasks import ClassificationTask
from classy_vision.trainer import LocalTrainer
from torchvision.transforms import ToTensor


# WARNING: The goal of this test is to use our public API as advertised in our
# tutorials and make sure everything trains successfully. If you break this
# test, make sure you also update our tutorials.


@register_loss("my_loss")
class MyLoss(ClassyLoss):
    def forward(self, input, target):
        labels = F.one_hot(target, num_classes=2).float()
        return F.binary_cross_entropy(input, labels)

    @classmethod
    def from_config(cls, config):
        # We don't need anything from the config
        return cls()


@register_dataset("my_dataset")
class MyDataset(SyntheticImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
            transform=ApplyTransformToKey(ToTensor()),
            num_samples=100,
            crop_size=224,
            class_ratio=0.5,
            seed=0,
        )

        test_dataset = MyDataset(
            batchsize_per_replica=32,
            shuffle=False,
            transform=ApplyTransformToKey(ToTensor()),
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
