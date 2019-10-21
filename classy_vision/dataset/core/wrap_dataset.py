#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import DataLoader

from .dataset import Dataset


class WrapDataset(Dataset):
    """
        Wraps a PyTorch dataset into our core dataset interface.
    """

    def __init__(self, dataset, key="input"):
        import torch.utils.data

        assert isinstance(dataset, torch.utils.data.Dataset)
        super(WrapDataset, self).__init__()
        self.dataset = dataset
        self.key = key

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            return sample

        return {self.key: sample}

    def __len__(self):
        return len(self.dataset)

    def iterator(self, *args, **kwargs):
        # WrapDataset will be removed soon
        return DataLoader(
            self,
            batch_size=kwargs.get("batchsize_per_replica", 1),
            num_workers=kwargs.get("num_workers", 0),
            pin_memory=kwargs.get("pin_memory", False),
            multiprocessing_context=kwargs.get("multiprocessing_context", None),
        )

    def get_classy_state(self):
        """Pytorch datasets don't have state"""
        return {
            # For debugging saved states
            "state": {"dataset_type": type(self)}
        }

    def set_classy_state(self, state):
        """Pytorch datasets don't have state"""
        return self
