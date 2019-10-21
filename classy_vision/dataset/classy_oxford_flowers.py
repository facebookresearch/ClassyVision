#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from . import register_dataset
from .classy_dataset import ClassyDataset
from .core import WrapDataset
from .transforms.util import TupleToMapTransform, build_field_transform_default_imagenet


# constants for Oxford Flowers dataset:
CANDIDATE_PATHS = ["/mnt/fair-flash-east/oxford102flowers.img"]
CLASS_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen ",
    "watercress",
    "canna lily",
    "hippeastrum ",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]
NUM_CLASSES = len(CLASS_NAMES)


@register_dataset("oxford_flowers")
class OxfordFlowersDataset(ClassyDataset):
    def __init__(self, split, batchsize_per_replica, shuffle, transform, num_samples):
        super().__init__(split, batchsize_per_replica, shuffle, transform, num_samples)
        self.dataset = self._load_dataset()
        self.dataset = self.wrap_dataset(
            self.dataset,
            transform,
            batchsize_per_replica,
            shuffle=shuffle,
            subsample=num_samples,
        )

    @classmethod
    def from_config(cls, config):
        assert "split" in config
        split = config["split"]
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        transform = build_field_transform_default_imagenet(
            transform_config, split=split
        )
        return cls(split, batchsize_per_replica, shuffle, transform, num_samples)

    def _load_dataset(self):
        # find location of images:
        img_dir = None
        for path in CANDIDATE_PATHS:
            if os.path.isdir(path):
                img_dir = path
                break
        assert img_dir is not None, "Oxford Flowers folder not found"
        img_dir = os.path.join(img_dir, self.split)

        # return flowers dataset:
        dataset = datasets.ImageFolder(img_dir)
        dataset = WrapDataset(dataset)
        self.transform = transforms.Compose(
            [TupleToMapTransform(["input", "target"]), self.transform]
        )
        return dataset

    def get_target_names(self):
        return CLASS_NAMES
