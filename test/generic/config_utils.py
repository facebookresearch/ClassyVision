#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from classy_vision.tasks import build_task


def get_test_task_config(head_num_classes=1000):
    return {
        "name": "classification_task",
        "num_epochs": 12,
        "loss": {"name": "CrossEntropyLoss"},
        "dataset": {
            "train": {
                "name": "synthetic_image",
                "crop_size": 224,
                "class_ratio": 0.5,
                "num_samples": 2000,
                "seed": 0,
                "batchsize_per_replica": 32,
                "use_shuffle": True,
                "transforms": [
                    {
                        "name": "apply_transform_to_key",
                        "transforms": [
                            {"name": "ToTensor"},
                            {
                                "name": "Normalize",
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        ],
                        "key": "input",
                    }
                ],
            },
            "test": {
                "name": "synthetic_image",
                "crop_size": 224,
                "class_ratio": 0.5,
                "num_samples": 2000,
                "seed": 0,
                "batchsize_per_replica": 32,
                "use_shuffle": False,
                "transforms": [
                    {
                        "name": "apply_transform_to_key",
                        "transforms": [
                            {"name": "ToTensor"},
                            {
                                "name": "Normalize",
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        ],
                        "key": "input",
                    }
                ],
            },
        },
        "meters": {"accuracy": {"topk": [1, 5]}},
        "model": {
            "name": "resnet",
            "num_blocks": [3, 4, 6, 3],
            "small_input": False,
            "zero_init_bn_residuals": True,
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
            "param_schedulers": {"lr": {"name": "step", "values": [0.1, 0.01]}},
            "weight_decay": 1e-4,
            "momentum": 0.9,
        },
    }


def get_fast_test_task_config(head_num_classes=1000):
    return {
        "name": "classification_task",
        "num_epochs": 1,
        "loss": {"name": "CrossEntropyLoss"},
        "dataset": {
            "train": {
                "name": "synthetic_image",
                "crop_size": 20,
                "class_ratio": 0.5,
                "num_samples": 10,
                "seed": 0,
                "batchsize_per_replica": 2,
                "use_shuffle": False,
                "transforms": [
                    {
                        "name": "apply_transform_to_key",
                        "transforms": [
                            {"name": "ToTensor"},
                            {
                                "name": "Normalize",
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        ],
                        "key": "input",
                    }
                ],
            },
            "test": {
                "name": "synthetic_image",
                "crop_size": 20,
                "class_ratio": 0.5,
                "num_samples": 10,
                "seed": 0,
                "batchsize_per_replica": 2,
                "use_shuffle": False,
                "transforms": [
                    {
                        "name": "apply_transform_to_key",
                        "transforms": [
                            {"name": "ToTensor"},
                            {
                                "name": "Normalize",
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        ],
                        "key": "input",
                    }
                ],
            },
        },
        "model": {
            "name": "resnet",
            "num_blocks": [1],
            "small_input": False,
            "zero_init_bn_residuals": True,
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
        "optimizer": {"name": "sgd", "lr": 0.01, "weight_decay": 1e-4, "momentum": 0.9},
    }


def get_test_classy_task():
    config = get_test_task_config()
    task = build_task(config)
    return task


def get_test_mlp_task_config():
    return {
        "name": "classification_task",
        "num_epochs": 10,
        "loss": {"name": "CrossEntropyLoss"},
        "dataset": {
            "train": {
                "name": "synthetic_image",
                "num_classes": 2,
                "crop_size": 20,
                "class_ratio": 0.5,
                "num_samples": 10,
                "seed": 0,
                "batchsize_per_replica": 3,
                "use_augmentation": False,
                "use_shuffle": True,
                "transforms": [
                    {
                        "name": "apply_transform_to_key",
                        "transforms": [
                            {"name": "ToTensor"},
                            {
                                "name": "Normalize",
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        ],
                        "key": "input",
                    }
                ],
            },
            "test": {
                "name": "synthetic_image",
                "num_classes": 2,
                "crop_size": 20,
                "class_ratio": 0.5,
                "num_samples": 10,
                "seed": 0,
                "batchsize_per_replica": 1,
                "use_augmentation": False,
                "use_shuffle": False,
                "transforms": [
                    {
                        "name": "apply_transform_to_key",
                        "transforms": [
                            {"name": "ToTensor"},
                            {
                                "name": "Normalize",
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        ],
                        "key": "input",
                    }
                ],
            },
        },
        "model": {
            "name": "mlp",
            # 3x20x20 = 1200
            "input_dim": 1200,
            "output_dim": 1000,
            "hidden_dims": [10],
        },
        "meters": {"accuracy": {"topk": [1]}},
        "optimizer": {
            "name": "sgd",
            "num_epochs": 10,
            "lr": 0.1,
            "weight_decay": 1e-4,
            "momentum": 0.9,
        },
    }


def get_test_model_configs():
    return [
        # resnet 50
        {
            "name": "resnet",
            "num_blocks": [3, 4, 6, 3],
            "small_input": False,
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
        # resnet 101
        {
            "name": "resnet",
            "num_blocks": [3, 4, 23, 3],
            "small_input": False,
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
        # resnext 101 32x4d
        {
            "name": "resnext",
            "num_blocks": [3, 4, 23, 3],
            "base_width_and_cardinality": [4, 32],
            "small_input": False,
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
    ]


def get_test_video_task_config():
    return {
        "name": "classification_task",
        "num_epochs": 27,
        "loss": {"name": "CrossEntropyLoss"},
        "dataset": {
            "train": {
                "name": "synthetic_video",
                "split": "train",
                "batchsize_per_replica": 8,
                "use_shuffle": True,
                "num_samples": 128,
                "frames_per_clip": 8,
                "video_height": 128,
                "video_width": 160,
                "num_classes": 50,
                "clips_per_video": 1,
            },
            "test": {
                "name": "synthetic_video",
                "split": "test",
                "batchsize_per_replica": 10,
                "use_shuffle": False,
                "num_samples": 40,
                "frames_per_clip": 8,
                "video_height": 128,
                "video_width": 160,
                "num_classes": 50,
                "clips_per_video": 10,
            },
        },
        "meters": {"accuracy": {"topk": [1, 5]}},
        "model": {
            "name": "resnext3d",
            "frames_per_clip": 8,
            "input_planes": 3,
            "clip_crop_size": 224,
            "skip_transformation_type": "postactivated_shortcut",
            "residual_transformation_type": "postactivated_bottleneck_transformation",
            "num_blocks": [3, 4, 6, 3],
            "input_key": "video",
            "stem_name": "resnext3d_stem",
            "stem_planes": 64,
            "stem_temporal_kernel": 5,
            "stem_spatial_kernel": 7,
            "stem_maxpool": True,
            "stage_planes": 64,
            "stage_temporal_kernel_basis": [[3], [3, 1], [3, 1], [1, 3]],
            "temporal_conv_1x1": [True, True, True, True],
            "stage_temporal_stride": [1, 1, 1, 1],
            "stage_spatial_stride": [1, 2, 2, 2],
            "num_groups": 1,
            "width_per_group": 64,
            "num_classes": 50,
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "default_head",
                    "pool_size": [8, 7, 7],
                    "activation_func": "softmax",
                    "num_classes": 50,
                    "fork_block": "pathway0-stage4-block2",
                    "in_plane": 512,
                    "use_dropout": True,
                }
            ],
        },
        "optimizer": {
            "name": "sgd",
            "param_schedulers": {
                "lr": {
                    "name": "multistep",
                    "num_epochs": 10,
                    "values": [0.1, 0.01, 0.001, 0.0001],
                    "milestones": [3, 7, 9],
                }
            },
            "weight_decay": 0.0001,
            "momentum": 0.9,
        },
    }


def get_test_classy_video_task():
    config = get_test_video_task_config()
    task = build_task(config)
    return task
