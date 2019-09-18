#!/usr/bin/env python3

import torch

from .dataset import Dataset


class ResampleDataset(Dataset):
    """
        Dataset that resamples a PyTorch dataset using either a sampling closure
        or a sampling list.
    """

    def __init__(self, dataset, resample, size=None):

        # assertions:
        assert isinstance(dataset, Dataset)
        if torch.is_tensor(resample):
            resample = resample.tolist()
        assert callable(resample) or isinstance(resample, list)
        if callable(resample):
            assert size is not None, "must provide size of resample closure"
        elif size is not None:
            assert size == len(resample), "specified size does not match list"
        else:
            size = len(resample)

        # create object:
        super(ResampleDataset, self).__init__()
        self.dataset = dataset
        self._resample = resample
        self.size = size

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self)
        if callable(self._resample):
            idx = self._resample(idx)
        else:
            idx = self._resample[idx]
        assert idx >= 0 and idx < len(
            self.dataset
        ), "Invalid index {i} for dataset of size {s}".format(
            i=idx, s=len(self.dataset)
        )
        return self.dataset[idx]

    def __len__(self):
        return self.size

    def get_classy_state(self):
        return {
            "state": {
                "dataset_type": type(self),
                "_resample": self._resample,
                "size": self.size,
            },
            "wrapped_state": self.dataset.get_classy_state(),
        }

    def set_classy_state(self, state):
        self._resample = state["state"]["_resample"]
        self.size = state["state"]["size"]
        self.dataset.set_classy_state(state["wrapped_state"])
        return self
