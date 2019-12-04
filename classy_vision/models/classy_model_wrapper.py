#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

from .classy_model import ClassyModel


class ClassyModelWrapper(ClassyModel):
    """
    Class which wraps an `nn.Module <https://pytorch.org/docs/stable/
    nn.html#torch.nn.Module>`_ within a ClassyModel.

    The only required argument is the model, the additional args are needed
    to get some additional capabilities from Classy Vision to work.
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple] = None,
        output_shape: Optional[Tuple] = None,
        model_depth: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._model_depth = model_depth

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        if hasattr(self.model, "extract_features"):
            return self.model.extract_features(x)
        return super().extract_features(x)

    @property
    def input_shape(self):
        if self._input_shape is not None:
            return self._input_shape
        return super().input_shape

    @property
    def output_shape(self):
        if self._output_shape is not None:
            return self._output_shape
        return super().output_shape

    @property
    def model_depth(self):
        if self._model_depth is not None:
            return self._model_depth
        return super().model_depth
