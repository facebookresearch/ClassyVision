#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile
import unittest
from test.generic.config_utils import get_test_task_config

import torch
from classy_vision.dataset.transforms import ClassyTransform
from classy_vision.hub import ClassyHubInterface
from classy_vision.models import ClassyModel, build_model
from classy_vision.tasks import ClassyTask, build_task
from torchvision import models, transforms


class TestTransform(ClassyTransform):
    def __call__(self, x):
        return x


class TestClassyHubInterface(unittest.TestCase):
    def setUp(self):
        # create a base directory to write image files to
        self.base_dir = tempfile.mkdtemp()
        self.image_path = self.base_dir + "/img.jpg"
        # create an image with a non standard size
        image_tensor = torch.zeros((3, 1000, 2500), dtype=torch.float)
        transforms.ToPILImage()(image_tensor).save(self.image_path)

    def tearDown(self):
        # delete all the temporary data created
        shutil.rmtree(self.base_dir)

    def _test_predict_and_extract_features(self, hub_interface: ClassyHubInterface):
        dataset = hub_interface.create_image_dataset(
            [self.image_path], phase_type="test"
        )
        data_iterator = hub_interface.get_data_iterator(dataset)
        input = next(data_iterator)
        # set the model to eval mode
        hub_interface.eval()
        output = hub_interface.predict(input)
        self.assertIsNotNone(output)
        # see the prediction for the input
        hub_interface.predict(input).argmax().item()
        # check extract features
        output = hub_interface.extract_features(input)
        self.assertIsNotNone(output)

    def _get_classy_model(self):
        config = get_test_task_config()
        model_config = config["model"]
        return build_model(model_config)

    def _get_non_classy_model(self):
        return models.resnet18(pretrained=False)

    def test_from_task(self):
        config = get_test_task_config()
        task = build_task(config)
        hub_interface = ClassyHubInterface.from_task(task)

        self.assertIsInstance(hub_interface.task, ClassyTask)
        self.assertIsInstance(hub_interface.model, ClassyModel)

        # this will pick up the transform from the task's config
        self._test_predict_and_extract_features(hub_interface)

        # test that the correct transform is picked up
        phase_type = "test"
        test_transform = TestTransform()
        task.datasets[phase_type].transform = test_transform
        hub_interface = ClassyHubInterface.from_task(task)
        dataset = hub_interface.create_image_dataset(
            [self.image_path], phase_type=phase_type
        )
        self.assertIsInstance(dataset.transform, TestTransform)

    def test_from_model(self):
        for model in [self._get_classy_model(), self._get_non_classy_model()]:
            hub_interface = ClassyHubInterface.from_model(model)

            self.assertIsNone(hub_interface.task)
            self.assertIsInstance(hub_interface.model, ClassyModel)

            # this will pick up the transform from imagenet
            self._test_predict_and_extract_features(hub_interface)
