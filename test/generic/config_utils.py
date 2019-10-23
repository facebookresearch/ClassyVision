#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from classy_vision.tasks import build_task

from .utils import Arguments


def get_test_task_config(head_num_classes=1000):
    return {
        "name": "classy_vision",
        "num_epochs": 12,
        "loss": {"name": "CrossEntropyLoss"},
        "dataset": {
            "train": {
                "name": "synthetic_image",
                "split": "train",
                "crop_size": 224,
                "class_ratio": 0.5,
                "num_samples": 2000,
                "seed": 0,
                "batchsize_per_replica": 32,
                "use_shuffle": True,
            },
            "test": {
                "name": "synthetic_image",
                "split": "test",
                "crop_size": 224,
                "class_ratio": 0.5,
                "num_samples": 2000,
                "seed": 0,
                "batchsize_per_replica": 32,
                "use_shuffle": False,
            },
        },
        "meters": {"accuracy": {"topk": [1, 5]}},
        "model": {
            "name": "resnet",
            "num_blocks": [3, 4, 6, 3],
            "small_input": False,
            "zero_init_bn_residuals": True,
            "freeze_trunk": False,
            "heads": [
                {
                    "name": "fully_connected",
                    "unique_id": "default_head",
                    "num_classes": head_num_classes,
                    "fork_block": "block3-2",
                    "in_plane": 2048,
                }
            ],
        },
        "optimizer": {
            "name": "sgd",
            "num_epochs": 12,
            "lr": {"name": "step", "values": [0.1, 0.01]},
            "weight_decay": 1e-4,
            "weight_decay_batchnorm": 0.0,
            "momentum": 0.9,
        },
    }


def get_fast_test_task_config(head_num_classes=1000):
    return {
        "name": "classy_vision",
        "num_epochs": 1,
        "loss": {"name": "CrossEntropyLoss"},
        "dataset": {
            "train": {
                "name": "synthetic_image",
                "split": "train",
                "crop_size": 20,
                "class_ratio": 0.5,
                "num_samples": 10,
                "seed": 0,
                "batchsize_per_replica": 2,
                "use_shuffle": False,
            },
            "test": {
                "name": "synthetic_image",
                "split": "test",
                "crop_size": 20,
                "class_ratio": 0.5,
                "num_samples": 10,
                "seed": 0,
                "batchsize_per_replica": 2,
                "use_shuffle": False,
            },
        },
        "model": {
            "name": "resnet",
            "num_blocks": [1],
            "small_input": False,
            "zero_init_bn_residuals": True,
            "freeze_trunk": False,
            "heads": [
                {
                    "name": "fully_connected",
                    "unique_id": "default_head",
                    "num_classes": head_num_classes,
                    "fork_block": "block0-0",
                    "in_plane": 256,
                }
            ],
        },
        "meters": {"accuracy": {"topk": [1]}},
        "optimizer": {
            "name": "sgd",
            "lr": 0.01,
            "weight_decay": 1e-4,
            "weight_decay_batchnorm": 0.0,
            "momentum": 0.9,
        },
    }


def get_test_args():
    return Arguments(device="cpu", num_workers=8, test_only=False)


def get_test_classy_task():
    config = get_test_task_config()
    args = get_test_args()
    task = build_task(config, args)
    return task


def get_test_model_configs():
    return [
        # resnet 101
        {
            "name": "resnet",
            "num_blocks": [3, 4, 6, 3],
            "small_input": False,
            "freeze_trunk": False,
            "heads": [
                {
                    "name": "fully_connected",
                    "unique_id": "default_head",
                    "num_classes": 1000,
                    "fork_block": "block3-2",
                    "in_plane": 2048,
                }
            ],
        },
        # resnext 101 32-4d
        {
            "name": "resnext",
            "num_blocks": [3, 4, 6, 3],
            "base_width_and_cardinality": [4, 32],
            "small_input": False,
            "freeze_trunk": False,
            "heads": [
                {
                    "name": "fully_connected",
                    "unique_id": "default_head",
                    "num_classes": 1000,
                    "fork_block": "block3-2",
                    "in_plane": 2048,
                }
            ],
        },
        # vgg 19
        {
            "name": "vgg",
            "num_classes": 1000,
            "depth": 19,
            "num_stages": 5,
            "stride2_inds": [],
            "max_pool_inds": [0, 1, 2, 3, 4],
            "ds_mult": 1.0,
            "ws_mult": 1.0,
            "bn_epsilon": 1e-5,
            "bn_momentum": 0.1,
            "relu_inplace": True,
            "small_input": False,
        },
    ]
