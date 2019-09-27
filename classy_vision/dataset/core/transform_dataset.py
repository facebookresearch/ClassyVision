#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .dataset import Dataset


class TransformDataset(Dataset):
    """
        Dataset that transforms a sample from a PyTorch dataset. The `transform`
        used as input can be a single transform function of a list of functions
        to be composed.
    """

    def __init__(self, dataset, transform):
        super(TransformDataset, self).__init__()
        assert isinstance(dataset, Dataset)
        if not isinstance(transform, list):
            transform = [transform]
        assert all(callable(t) for t in transform)
        self.dataset = dataset
        self._transform = transform

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self.dataset)
        sample = self.dataset[idx]
        for transform in self._transform:
            sample = transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)
