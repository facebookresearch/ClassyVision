#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .dataset import Dataset


class WrapTorchVisionVideoDataset(Dataset):
    """
        Wraps a TorchVision video dataset into our core dataset interface.
        A video dataset can contain both video and audio data
    """

    def __init__(self, dataset):
        import torch.utils.data

        assert isinstance(dataset, torch.utils.data.Dataset)
        super(WrapTorchVisionVideoDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        video, audio, target = self.dataset[idx]
        return {"input": {"video": video, "audio": audio}, "target": target}

    def __len__(self):
        return len(self.dataset)

    def get_classy_state(self):
        # Pytorch datasets don't have state
        return {
            # For debugging saved states
            "state": {"dataset_type": type(self)}
        }
