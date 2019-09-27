#!/usr/bin/env python3

from .dataset import Dataset


class WrapTorchVisionVideoDataset(Dataset):
    """
        Wraps a TorchVision video dataset into our core dataset interface.
        A videp dataset can contain both video and audio data
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

    def set_classy_state(self, state):
        # Pytorch datasets don't have state
        return self
