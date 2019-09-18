#!/usr/bin/env python3

import torch.nn as nn


class ClassyVisionHead(nn.Module):
    def __init__(self, head_config):
        """
        Classy Head constructor. This stores the head config for future access.
        This is also the place to build and initialize the layers.
        """
        assert "name" in head_config
        assert "unique_id" in head_config
        super().__init__()
        self._config = head_config

    @property
    def unique_id(self):
        """
        return a global unique identifier for the head.
        """
        return self._config["unique_id"]

    def forward(self, x):
        raise NotImplementedError
