#!/usr/bin/env python3

import os

import torch.utils.data as data
from torchvision.datasets.folder import default_loader, is_image_file


# constants for the KITTI dataset:
DATA_PATH = "/mnt/fair-flash-east/kitti2015.img"
# TODO: Create validation partition.


# helper function that returns list of images in folder:
def _find_images(folder):
    images = []
    for root, _, fnames in sorted(os.walk(folder)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                images.append(os.path.join(root, fname))
    return images


# class for KITTI dataset:
class KittiDataset(data.Dataset):
    def __init__(self, root, split, transform=None, loader=default_loader):

        # assertions:
        assert os.path.isdir(root), "folder %s not found" % root
        self.transform = transform
        self.loader = loader

        # find image folder:
        folder = os.path.join(root, "training" if split == "train" else "testing")
        assert os.path.isdir(folder), "folder %s not found" % folder

        # make lists of images (and corresponding targets):
        all_imgs = _find_images(os.path.join(folder, "image_2"))
        self.imgs1 = [img for img in all_imgs if "_10." in img]  # left image
        self.imgs2 = [img for img in all_imgs if "_11." in img]  # right image
        self.targets = (
            None
            if split == "test"
            else _find_images(os.path.join(folder, "disp_noc_0"))
        )
        assert len(self.imgs1) == len(self.imgs2)
        if self.targets is not None:
            assert len(self.imgs1) == len(self.targets)

    def __getitem__(self, idx):
        img1 = self.loader(self.imgs1[idx])
        img2 = self.loader(self.imgs2[idx])
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.targets is None:
            return (img1, img2)
        else:
            target = self.loader(self.targets[idx])
            return ((img1, img2), target)

    def __len__(self):
        return len(self.imgs1)


# function that loads the KITTI dataset:
def get_dataset(split):
    avail_splits = get_avail_splits()
    assert split in avail_splits
    return KittiDataset(DATA_PATH, split), None


def get_class_names(split=None):
    raise NotImplementedError()


def get_avail_splits():
    return ["train", "test"]


# This returns two disjoint sets with images and labels
# Note: This may NOT be the official train or test split of the dataset
def get_default_train_test_splits():
    return {"train": "train", "test": "test"}
