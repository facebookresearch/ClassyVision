#!/usr/bin/env python3

from classy_vision.tasks import setup_task

from .utils import Arguments


def get_test_task_config():
    return {
        "name": "classy_vision",
        "num_phases": 12,
        "criterion": {"name": "sum_cross_entropy"},
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
                    "name": "resnext_fc",
                    "unique_id": "default_head",
                    "num_classes": 1000,
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


def get_test_args():
    return Arguments(device="cpu", num_workers=8, test_only=False)


def get_test_classy_task():
    config = get_test_task_config()
    args = get_test_args()
    task = setup_task(config, args, local_rank=0)
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
                    "name": "resnext_fc",
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
                    "name": "resnext_fc",
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
