#!/usr/bin/env python3

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

    def get_classy_state(self):
        """Pytorch datasets don't have state"""
        return {
            # For debugging saved states
            "state": {"dataset_type": type(self)}
        }

    def set_classy_state(self, state):
        """Pytorch datasets don't have state"""
        return self
