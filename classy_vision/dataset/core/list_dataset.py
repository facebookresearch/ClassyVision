#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets.folder import default_loader


class ListDataset:
    """Dataset that loads data using a list of items, a corresponding loader,
    and a list of metadata. The default loader is an image file loader so this
    dataset can be used directly with a list of image files.
    You can use it without metadata if you set metadata parameter to None
    """

    def __init__(self, files, metadata=None, loader=default_loader):
        """
        metadata (List[Dict[Type]] or List[Type], Optional):
            metadata to be added to each sample.
            The Type can be anything that pytorch default_collate can handle.
            If Type is tensor, make sure that the tensors are of same dimension.
        """
        if metadata is not None:
            assert isinstance(metadata, list), "metadata should be a list"
            assert len(files) == len(metadata)
            assert len(files) > 0, "Empty ListDataset is not allowed"
            if not isinstance(metadata[0], dict):
                metadata = [{"target": target} for target in metadata]
        self.files = files
        self.metadata = metadata
        self.loader = loader

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self)
        img = self.loader(self.files[idx])
        item = {"input": img}
        if self.metadata is not None:
            item.update(self.metadata[idx])

        return item

    def __len__(self):
        return len(self.files)
