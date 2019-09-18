#!/usr/bin/env python3
import logging

import torch

from ...generic.util import torch_seed
from .dataset import Dataset
from .resample_dataset import ResampleDataset


class ShuffleDataset(ResampleDataset):
    """
        Dataset that shuffles a dataset.
    """

    def __init__(self, dataset, seed=0):
        assert isinstance(
            dataset, Dataset
        ), f"'dataset' of type '{type(dataset)}' is not an instance of 'Dataset'"
        assert isinstance(
            seed, int
        ), f"Unsupported value {seed} for 'seed', should be an int"
        self._seed = seed
        resample = [idx for idx in range(len(dataset))]
        super(ShuffleDataset, self).__init__(dataset, resample)

    def do_shuffle(self, epoch_num=None, size=None):
        """Shuffle the data for the specified epoch and then restore the PRNG
        state at the end."""
        logging.info("Do actual data shuffling in ShuffleDataset")
        assert isinstance(
            epoch_num, int
        ), f"Unsupported value {epoch_num} for 'epoch_num', should be an int"
        if size is None:
            size = len(self.dataset)
        self.size = size
        seed = self._seed + epoch_num
        with torch_seed(seed):
            self._resample = torch.randperm(len(self.dataset)).tolist()[:size]

    def get_classy_state(self):
        state = super().get_classy_state()
        state["state"]["_seed"] = self._seed
        return state

    def set_classy_state(self, state):
        super().set_classy_state(state)
        self._seed = state["state"]["_seed"]
        return self
