#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import sys
import unittest

import classy_vision
from classy_vision.dataset.transforms import GenericImageTransform
from classy_vision.optim import SGD
from classy_vision.optim.param_scheduler import LinearParamScheduler
from classy_vision.tasks import ClassificationTask
from classy_vision.trainer import LocalTrainer
from torchvision import transforms


# import the classes from the synthetic template
path = pathlib.Path(classy_vision.__file__).resolve().parent
synthetic_template_path = path / "templates" / "synthetic"
sys.path.append(str(synthetic_template_path))

from datasets.my_dataset import MyDataset  # isort:skip
from losses.my_loss import MyLoss  # isort:skip
from models.my_model import MyModel  # isort:skip


# WARNING: The goal of this test is to use our public API as advertised in our
# tutorials and make sure everything trains successfully. If you break this
# test, make sure you also update our tutorials.


class APITest(unittest.TestCase):
    def test_one(self):
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

        optimizer = SGD(momentum=0.9, weight_decay=1e-4, nesterov=True)
        optimizer.set_param_schedulers(
            {"lr": LinearParamScheduler(start_lr=0.01, end_lr=0.009)}
        )

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
