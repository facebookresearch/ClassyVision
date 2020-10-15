#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from classy_vision.dataset import ClassyDataset
from classy_vision.dataset.image_path_dataset import ImagePathDataset
from classy_vision.dataset.transforms import ClassyTransform
from classy_vision.dataset.transforms.util import build_field_transform_default_imagenet
from classy_vision.models import ClassyModel
from classy_vision.tasks import ClassyTask


class ClassyHubInterface:
    """PyTorch Hub interface for classy vision tasks and models.

    The task is optional, but a model is guaranteed to be present.  Do
    not use the constructor directly, instead Use from_task() or
    from_model() to instantiate the class.

    See the examples folder for an example of how to use this class

    Attributes:
        task: If present, task that can be used to train the torchhub model
        model: torchub model

    """

    def __init__(
        self, task: Optional[ClassyTask] = None, model: Optional[ClassyModel] = None
    ) -> None:
        """Constructor for ClassyHubInterface.

        Only one of task or model can be specified at construction
        time. If task is specified then task.model is used to populate
        the model attribute.

        Do not use the constructor directly, instead use from_task()
        or from_model() to instantiate the class.

        Args:
            task: task that can be used to train torchhub model,
                task.model is used to populate the model attribute
            model: torchhub model
        """
        self.task = task
        if task is None:
            assert model is not None, "Need to specify a model if task is None"
            self.model = model
        else:
            assert model is None, "Cannot pass a model if task is not None"
            self.model = task.model

    @classmethod
    def from_task(cls, task: ClassyTask) -> "ClassyHubInterface":
        """Instantiates the ClassyHubInterface from a task.

        This function returns a hub interface based on a ClassyTask.

        Args:
            task: ClassyTask that contains hub model

        """
        return cls(task=task)

    @classmethod
    def from_model(cls, model: Union[nn.Module, ClassyModel]) -> "ClassyHubInterface":
        """Instantiates the ClassyHubInterface from a model.

        This function returns a hub interface based on a ClassyModel

        Args:
            model: torchhub model

        """
        if not isinstance(model, ClassyModel):
            model = ClassyModel.from_model(model)
        return cls(model=model)

    def create_image_dataset(
        self,
        batchsize_per_replica: int = 32,
        shuffle: bool = True,
        transform: Optional[Union[ClassyTransform, Callable]] = None,
        num_samples: Optional[int] = None,
        image_folder: Optional[str] = None,
        image_files: Optional[List[str]] = None,
        phase_type: str = "train",
    ) -> ClassyDataset:
        """Create a ClassyDataset which reads images from image_paths.

        See :class:`dataset.ImagePathDataset` for documentation on image_folder and
        image_files

        Args:
            batchsize_per_replica: Minibatch size per replica (i.e. samples per GPU)
            shuffle: If true, data is shuffled between epochs
            transform: Transform to apply to sample. If left as None, the dataset's
                phase_type is used to determine the transform to apply. The transform
                for the phase_type is searched for in self.task, falling back to
                imagenet transformations if it is not found there.
            num_samples: If specified, limits the number of samples returned by
                the dataset
            phase_type: String specifying the phase_type, e.g. "train" or "test"
        """
        if transform is None:
            if self.task is not None and phase_type in self.task.datasets:
                # use the transform from the dataset for the phase_type
                dataset = self.task.datasets[phase_type]
                transform = dataset.transform
                assert transform is not None, "Cannot infer transform from the task"
            else:
                transform = build_field_transform_default_imagenet(
                    config=None, split=phase_type, key_map_transform=None
                )
        return ImagePathDataset(
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            image_folder=image_folder,
            image_files=image_files,
        )

    @staticmethod
    def get_data_iterator(dataset: ClassyDataset) -> Iterator[Any]:
        """Returns an iterator that can be used to retrieve training / testing samples.

        Args:
            dataset: Dataset to iterate over
        """
        return iter(dataset.iterator())

    def train(self) -> None:
        """Sets the model to train mode and enables torch gradient calculation"""
        torch.autograd.set_grad_enabled(True)
        self.model.train()

    def eval(self) -> None:
        """Sets the model to eval mode and disables torch gradient calculation"""
        torch.autograd.set_grad_enabled(False)
        self.model.eval()

    def predict(self, sample):
        """Returns the model's prediction for a sample.

        Args:
            sample: Must contain "input" key, model calculates prediction over input.
        """
        output = self.model(sample["input"])
        # squeeze the output in case the batch size is 1
        return output.squeeze()

    def extract_features(self, sample):
        """Calculates feature embeddings of sample.

        Args:
            sample: Must contain "input" key, model calculates prediction over input.
        """
        output = self.model.extract_features(sample["input"])
        # squeeze the output in case the batch size is 1
        return output.squeeze()
