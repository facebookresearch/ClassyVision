#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from classy_vision.dataset import ClassyDataset
from classy_vision.dataset.image_path_dataset import ImagePathDataset
from classy_vision.dataset.transforms import build_transforms
from classy_vision.models import ClassyVisionModel
from classy_vision.models.classy_model_wrapper import ClassyModelWrapper
from classy_vision.tasks import ClassyVisionTask


class ClassyHubInterface:
    """
    PyTorch Hub interface to classy vision tasks and models.

    The task is optional, but a model is guaranteed to be present.
    Use from_task() or from_model() to instantiate the class.
    """

    def __init__(
        self,
        task: Optional[ClassyVisionTask] = None,
        model: Optional[ClassyVisionModel] = None,
    ) -> None:
        self.task = task
        if task is None:
            assert model is not None, "Need to specify a model if task is None"
            self.model = model
        else:
            assert model is None, "Cannot pass a model if task is not None"
            self.model = task.model

    @classmethod
    def from_task(cls, task: ClassyVisionTask) -> "ClassyHubInterface":
        return cls(task=task)

    @classmethod
    def from_model(
        cls, model: Union[nn.Module, ClassyVisionModel]
    ) -> "ClassyHubInterface":
        if not isinstance(model, ClassyVisionModel):
            model = ClassyModelWrapper(model)
        return cls(model=model)

    def create_image_dataset(
        self,
        image_paths: Union[List[str], str],
        targets: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        default_transform: Optional[Callable] = None,
        split: str = "train",
    ) -> ClassyDataset:
        """
        Create a ClassyDataset which reads images from image_paths.


        image_paths: Can be
            - A single directory location, in which case the data is expected to be
                arranged in a format similar to torchvision.datasets.ImageFolder.
                The targets will be inferred from the directory structure.
            - A list of paths, in which case the list will contain the paths to all the
                images. In this situation, the targets can be specified by the targets
                argument.
        targets (optional): A list containing the target classes for each image
        default_transform (optional): Transform to apply if one isn't specified in the
            config. If left as None, the dataset's split is used to determine the
            transform to apply. The transform for the split is searched for in
            self.task, falling back to imagenet transformations if it is not found
            there.
        """
        if config is None:
            # create a default config
            config = {
                "batchsize_per_replica": 32,
                "use_shuffle": split == "train",
                "num_samples": None,
                "split": split,
            }
        if default_transform is None:
            if self.task is not None:
                # TODO (@mannatsingh): The transforms aren't picked up from
                # self.task's datasets, but from the task's config.
                transform_config = (
                    self.task.get_config()["dataset"]
                    .get(split, {})
                    .get("transforms", None)
                )
                if transform_config is not None:
                    default_transform = build_transforms(transform_config)
        return ImagePathDataset(
            config,
            image_paths=image_paths,
            targets=targets,
            default_transform=default_transform,
            split=split,
        )

    @staticmethod
    def get_data_iterator(dataset: ClassyDataset) -> Iterator[Any]:
        return iter(dataset.iterator())

    def train(self) -> None:
        """
        Sets the model to train mode and enables torch gradient calculation
        """
        torch.autograd.set_grad_enabled(True)
        self.model.train()

    def eval(self) -> None:
        """
        Sets the model to eval mode and disables torch gradient calculation
        """
        torch.autograd.set_grad_enabled(False)
        self.model.eval()

    def predict(self, sample):
        output = self.model(sample["input"])
        # squeeze the output in case the batch size is 1
        return output.squeeze()

    def extract_features(self, sample):
        output = self.model.extract_features(sample["input"])
        # squeeze the output in case the batch size is 1
        return output.squeeze()
