#!/usr/bin/env python3

import os

from classy_vision.dataset.core import ListDataset

from . import register_dataset
from .classy_dataset import ClassyDataset
from .transforms.util import build_field_transform_default_imagenet


# constants for Places dataset:
CANDIDATE_PATHS = [
    "/mnt/fair-flash-east/places_dataset.img",
    "/mnt/fair/places_dataset.img",
]
SPLIT_FOLDER = "/mnt/vol/gfsfblearner-oregon/users/dhruvm/places365_data/"
CLASS_FILE = (
    "/mnt/vol/gfsai-east/ai-group/users/imisra/datasets/places365/"
    "categories_places365.txt"
)
SPLIT_FILE = {
    "train": os.path.join(SPLIT_FOLDER, "places365_train_standard.txt"),
    "test": os.path.join(SPLIT_FOLDER, "val_large/path_and_label_val.txt"),
}  # TODO: Move test set from PRN to ASH.
NUM_CLASSES = 365


@register_dataset("places365")
class Places365Dataset(ClassyDataset):
    def __init__(self, config):
        super(Places365Dataset, self).__init__(config)
        # For memoizing target names
        self._target_names = None
        self.dataset = self._load_dataset()
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = self.parse_config(self._config)
        transform = build_field_transform_default_imagenet(
            transform_config, split=self._split
        )
        self.dataset = self.wrap_dataset(
            self.dataset,
            transform,
            batchsize_per_replica,
            shuffle=shuffle,
            subsample=num_samples,
        )

    def _load_dataset(self):
        # find location of images:
        img_dir = None
        for path in CANDIDATE_PATHS:
            if os.path.isdir(path):
                img_dir = path
                break
        assert img_dir is not None and os.path.isdir(img_dir), "Places folder not found"

        # load correct split:
        with open(SPLIT_FILE[self._split], "rt") as fread:
            split_char = " " if self._split == "train" else ","
            lines = [l.strip().split(split_char) for l in fread.readlines()]
            if self._split == "train":
                imgs = ["%s%s" % (img_dir, l[0]) for l in lines]
            else:
                imgs = [l[0] for l in lines]
        targets = [int(l[1]) for l in lines]

        # return dataset:
        return ListDataset(imgs, targets)

    def get_target_names(self):
        # Memoize target names
        if self._target_names is not None:
            return self._target_names

        class_names = []
        with open(CLASS_FILE, "r") as fh:
            for line in fh:
                # need to sanitize folder names
                cname, cid = line.strip().split()
                class_names.append(cname)
        self._target_names = class_names
        return self._target_names
