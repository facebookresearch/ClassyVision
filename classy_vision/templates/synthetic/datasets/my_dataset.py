#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from classy_vision.dataset import register_dataset
from classy_vision.dataset.classy_synthetic_image import SyntheticImageDataset


@register_dataset("my_dataset")
class MyDataset(SyntheticImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
