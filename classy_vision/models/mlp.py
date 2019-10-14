#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MLP model."""

import torch.nn as nn

from . import register_model
from .classy_vision_model import ClassyVisionModel


@register_model("mlp")
class MLP(ClassyVisionModel):
    """MLP model using ReLU. Useful for testing on CPUs."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        dropout,
        first_dropout,
        use_batchnorm,
        first_batchnorm,
        num_classes,
        freeze_trunk,
    ):
        super().__init__(num_classes, freeze_trunk)

        layers = []
        # If first_batchnorm is set, must be using batchnorm
        assert not first_batchnorm or use_batchnorm

        self._num_inputs = input_dim
        self._num_classes = output_dim
        self._model_depth = len(hidden_dims) + 1

        if dropout > 0 and first_dropout:
            layers.append(nn.Dropout(p=dropout))

        if use_batchnorm and first_batchnorm:
            layers.append(nn.BatchNorm1d(input_dim))

        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU(inplace=True))
            input_dim = dim

        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, config):
        assert (key in config for key in ["input_dim", "output_dim", "hidden_dims"])

        output_dim = config["output_dim"]
        return cls(
            input_dim=config["input_dim"],
            output_dim=output_dim,
            hidden_dims=config["hidden_dims"],
            dropout=config.get("dropout", 0),
            first_dropout=config.get("first_dropout", False),
            use_batchnorm=config.get("use_batchnorm", False),
            first_batchnorm=config.get("first_batchnorm", False),
            num_classes=output_dim,
            freeze_trunk=config.get("freeze_trunk", False),
        )

    def forward(self, x):
        batchsize_per_replica = x.shape[0]
        out = x.view(batchsize_per_replica, -1)
        out = self.mlp(out)
        return out

    @property
    def input_shape(self):
        return (self._num_inputs,)

    @property
    def output_shape(self):
        return (1, self._num_classes)

    @property
    def model_depth(self):
        return self._model_depth

    def validate(self, dataset_output_shape):
        return self.input_shape == dataset_output_shape
