#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from classy_vision.generic.util import is_pos_int, is_pos_int_list

from . import register_model
from .classy_model import ClassyModel
from .resnext3d_stage import ResStage
from .resnext3d_stem import R2Plus1DStem, ResNeXt3DStem


model_stems = {
    "r2plus1d_stem": R2Plus1DStem,
    "resnext3d_stem": ResNeXt3DStem,
    # For more types of model stem, add them below
}


class ResNeXt3DBase(ClassyModel):
    def __init__(
        self,
        input_key,
        input_planes,
        clip_crop_size,
        frames_per_clip,
        num_blocks,
        stem_name,
        stem_planes,
        stem_temporal_kernel,
        stem_spatial_kernel,
        stem_maxpool,
    ):
        """
        ResNeXt3DBase implements everything in ResNeXt3D model except the
        construction of 4 stages. See more details in ResNeXt3D.
        """
        super(ResNeXt3DBase, self).__init__()

        self._input_key = input_key
        self.input_planes = input_planes
        self.clip_crop_size = clip_crop_size
        self.frames_per_clip = frames_per_clip
        self.num_blocks = num_blocks

        assert stem_name in model_stems, "unknown stem: %s" % stem_name
        self.stem = model_stems[stem_name](
            stem_temporal_kernel,
            stem_spatial_kernel,
            input_planes,
            stem_planes,
            stem_maxpool,
        )

    @staticmethod
    def _parse_config(config):
        ret_config = {}
        required_args = [
            "input_planes",
            "clip_crop_size",
            "skip_transformation_type",
            "residual_transformation_type",
            "frames_per_clip",
            "num_blocks",
        ]
        for arg in required_args:
            assert arg in config, "resnext3d model requires argument %s" % arg
            ret_config[arg] = config[arg]

        # Default setting for model stem, which is considered as stage 0. Stage
        # index starts from 0 as implemented in ResStageBase._block_name() method.
        #   stem_planes: No. of output channles of conv op in stem
        #   stem_temporal_kernel: temporal size of conv op in stem
        #   stem_spatial_kernel: spatial size of conv op in stem
        #   stem_maxpool: by default, spatial maxpool op is disabled in stem
        ret_config.update(
            {
                "input_key": config.get("input_key", None),
                "stem_name": config.get("stem_name", "resnext3d_stem"),
                "stem_planes": config.get("stem_planes", 64),
                "stem_temporal_kernel": config.get("stem_temporal_kernel", 3),
                "stem_spatial_kernel": config.get("stem_spatial_kernel", 7),
                "stem_maxpool": config.get("stem_maxpool", False),
            }
        )
        # Default setting for model stages 1, 2, 3 and 4
        #   stage_planes: No. of output channel of 1st conv op in stage 1
        #   stage_temporal_kernel_basis: Basis of temporal kernel sizes for each of
        #       the stage.
        #   temporal_conv_1x1: if True, do temporal convolution in the fist
        #      1x1 Conv3d. Otherwise, do it in the second 3x3 Conv3d (default settting)
        #   stage_temporal_stride: temporal stride for each stage
        #   stage_spatial_stride: spatial stride for each stage
        #   num_groups: No. of groups in 2nd (group) conv in the residual transformation
        #   width_per_group: No. of channels per group in 2nd (group) conv in the
        #     residual transformation
        ret_config.update(
            {
                "stage_planes": config.get("stage_planes", 256),
                "stage_temporal_kernel_basis": config.get(
                    "stage_temporal_kernel_basis", [[3], [3], [3], [3]]
                ),
                "temporal_conv_1x1": config.get(
                    "temporal_conv_1x1", [False, False, False, False]
                ),
                "stage_temporal_stride": config.get(
                    "stage_temporal_stride", [1, 2, 2, 2]
                ),
                "stage_spatial_stride": config.get(
                    "stage_spatial_stride", [1, 2, 2, 2]
                ),
                "num_groups": config.get("num_groups", 1),
                "width_per_group": config.get("width_per_group", 64),
            }
        )
        # Default setting for model parameter initialization
        ret_config.update(
            {
                "zero_init_residual_transform": config.get(
                    "zero_init_residual_transform", False
                )
            }
        )
        assert is_pos_int_list(ret_config["num_blocks"])
        assert is_pos_int(ret_config["stem_planes"])
        assert is_pos_int(ret_config["stem_temporal_kernel"])
        assert is_pos_int(ret_config["stem_spatial_kernel"])
        assert type(ret_config["stem_maxpool"]) == bool
        assert is_pos_int(ret_config["stage_planes"])
        assert type(ret_config["stage_temporal_kernel_basis"]) == list
        assert all(
            is_pos_int_list(l) for l in ret_config["stage_temporal_kernel_basis"]
        )
        assert type(ret_config["temporal_conv_1x1"]) == list
        assert is_pos_int_list(ret_config["stage_temporal_stride"])
        assert is_pos_int_list(ret_config["stage_spatial_stride"])
        assert is_pos_int(ret_config["num_groups"])
        assert is_pos_int(ret_config["width_per_group"])
        return ret_config

    def _init_parameter(self, zero_init_residual_transform):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if (
                    hasattr(m, "final_transform_op")
                    and m.final_transform_op
                    and zero_init_residual_transform
                ):
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) and m.affine:
                if (
                    hasattr(m, "final_transform_op")
                    and m.final_transform_op
                    and zero_init_residual_transform
                ):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def set_classy_state(self, state, strict=True):
        # We need to support both regular checkpoint loading and 2D conv weight
        # inflation into 3D conv weight in this function.
        self.load_head_states(state)

        # clear the heads to set the trunk state
        attached_heads = self.get_heads()
        self.clear_heads()

        current_state = self.state_dict()
        for name, weight_src in state["model"]["trunk"].items():
            if name not in current_state:
                logging.warn(f"weight {name} is not found in current ResNeXt3D state")
                continue

            weight_tgt = current_state[name]
            assert (
                weight_src.dim() == weight_tgt.dim()
            ), "weight of source- and target 3D convolution should have same dimension"
            if (
                weight_src.dim() == 5
                and weight_src.shape[2] == 1
                and weight_tgt.shape[2] > 1
            ):
                # Find a source weight tensor where temporal dimension is 1. If the
                # temporal dimension of the current weight tensor with the same name
                # is larger than 1, we inflate the source weight tensor before
                # loading it. Such parameter inflation was first introduced in
                # the paper (https://arxiv.org/abs/1705.07750). It can achieve a
                # better initialization compared to random initialization.
                assert (
                    weight_src.shape[-2:] == weight_tgt.shape[-2:]
                    and weight_src.shape[:2] == weight_tgt.shape[:2]
                ), "weight shapes of source- and target 3D convolution mismatch"
                weight_src_inflated = (
                    weight_src.repeat(1, 1, weight_tgt.shape[2], 1, 1)
                    / weight_tgt.shape[2]
                )
                weight_src = weight_src_inflated
            else:
                assert all(
                    weight_src.size(d) == weight_tgt.size(d)
                    for d in range(weight_src.dim())
                ), "the shapes of source and target weight mismatch: %s Vs %s" % (
                    str(weight_src.size()),
                    str(weight_tgt.size()),
                )

            current_state[name] = weight_src.clone()
        self.load_state_dict(current_state, strict=strict)

        # set the heads back again
        self.set_heads(attached_heads)

    def forward(self, x):
        """
        Args:
            x (dict or torch.Tensor): video input.
                When its type is dict, the dataset is a video dataset, and its
                content is like {"video": torch.tensor, "audio": torch.tensor}.
                When its type is torch.Tensor, the dataset is an image dataset.
        """
        assert isinstance(x, dict) or isinstance(
            x, torch.Tensor
        ), "x must be either a dictionary or a torch.Tensor"
        if isinstance(x, dict):
            assert self._input_key is not None and self._input_key in x, (
                "input key (%s) not in the input" % self._input_key
            )
            x = x[self._input_key]
        else:
            assert (
                self._input_key is None
            ), "when input of forward pass is a tensor, input key should not be set"
            assert x.dim() == 4 or x.dim() == 5, "tensor x must be 4D/5D tensor"
            if x.dim() == 4:
                # x is a 4D tensor of size N x C x H x W and is prepared from an
                # image dataset. We insert a temporal axis make it 5D of size
                # N x C x T x H x W
                x = torch.unsqueeze(x, 2)

        out = self.stem([x])
        out = self.stages(out)

        return out

    @property
    def input_shape(self):
        """
        Shape of video model input can vary in the following cases
        - At training stage, input are video frame croppings of fixed size.
        - At test stage, input are original video frames to support Fully Convolutional
            evaluation and its size can vary video by video
        """
        # Input shape is used by tensorboard hook. We put the input shape at
        # training stage for profiling and visualization purpose.
        return (
            self.input_planes,
            self.frames_per_clip,
            self.clip_crop_size,
            self.clip_crop_size,
        )

    @property
    def input_key(self):
        return self._input_key


@register_model("resnext3d")
class ResNeXt3D(ResNeXt3DBase):
    """
    Implementation of:
        1. Conventional `post-activated 3D ResNe(X)t <https://arxiv.org/
        abs/1812.03982>`_.

        2. `Pre-activated 3D ResNe(X)t <https://arxiv.org/abs/1811.12814>`_.
        The model consists of one stem, a number of stages, and one or multiple
        heads that are attached to different blocks in the stage.
    """

    def __init__(
        self,
        input_key,
        input_planes,
        clip_crop_size,
        skip_transformation_type,
        residual_transformation_type,
        frames_per_clip,
        num_blocks,
        stem_name,
        stem_planes,
        stem_temporal_kernel,
        stem_spatial_kernel,
        stem_maxpool,
        stage_planes,
        stage_temporal_kernel_basis,
        temporal_conv_1x1,
        stage_temporal_stride,
        stage_spatial_stride,
        num_groups,
        width_per_group,
        zero_init_residual_transform,
    ):
        """
        Args:
            input_key (str): a key that can index into model input that is
                of dict type.
            input_planes (int): the channel dimension of the input. Normally 3 is used
                for rgb input.
            clip_crop_size (int): spatial cropping size of video clip at train time.
            skip_transformation_type (str): the type of skip transformation.
            residual_transformation_type (str): the type of residual transformation.
            frames_per_clip (int): Number of frames in a video clip.
            num_blocks (list): list of the number of blocks in stages.
            stem_name (str): name of model stem.
            stem_planes (int): the output dimension of the convolution in the model
                stem.
            stem_temporal_kernel (int): the temporal kernel size of the convolution
                in the model stem.
            stem_spatial_kernel (int): the spatial kernel size of the convolution
                in the model stem.
            stem_maxpool (bool): If true, perform max pooling.
            stage_planes (int): the output channel dimension of the 1st residual stage
            stage_temporal_kernel_basis (list): Basis of temporal kernel sizes for
                each of the stage.
            temporal_conv_1x1 (bool): Only useful for BottleneckTransformation.
                In a pathaway, if True, do temporal convolution in the first 1x1
                Conv3d. Otherwise, do it in the second 3x3 Conv3d.
            stage_temporal_stride (int): the temporal stride of the residual
                transformation.
            stage_spatial_stride (int): the spatial stride of the the residual
                transformation.
            num_groups (int): number of groups for the convolution.
                num_groups = 1 is for standard ResNet like networks, and
                num_groups > 1 is for ResNeXt like networks.
            width_per_group (int): Number of channels per group in 2nd (group)
                conv in the residual transformation in the first stage
            zero_init_residual_transform (bool): if true, the weight of last
                operation, which could be either BatchNorm3D in post-activated
                transformation or Conv3D in pre-activated transformation, in the
                residual transformation is initialized to zero
        """
        super(ResNeXt3D, self).__init__(
            input_key,
            input_planes,
            clip_crop_size,
            frames_per_clip,
            num_blocks,
            stem_name,
            stem_planes,
            stem_temporal_kernel,
            stem_spatial_kernel,
            stem_maxpool,
        )

        num_stages = len(num_blocks)
        out_planes = [stage_planes * 2 ** i for i in range(num_stages)]
        in_planes = [stem_planes] + out_planes[:-1]
        inner_planes = [
            num_groups * width_per_group * 2 ** i for i in range(num_stages)
        ]

        stages = []
        for s in range(num_stages):
            stage = ResStage(
                s + 1,  # stem is viewed as stage 0, and following stages start from 1
                [in_planes[s]],
                [out_planes[s]],
                [inner_planes[s]],
                [stage_temporal_kernel_basis[s]],
                [temporal_conv_1x1[s]],
                [stage_temporal_stride[s]],
                [stage_spatial_stride[s]],
                [num_blocks[s]],
                [num_groups],
                skip_transformation_type,
                residual_transformation_type,
                disable_pre_activation=(s == 0),
                final_stage=(s == (num_stages - 1)),
            )
            stages.append(stage)

        self.stages = nn.Sequential(*stages)
        self._init_parameter(zero_init_residual_transform)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResNeXt3D":
        """Instantiates a ResNeXt3D from a configuration.

        Args:
            config: A configuration for a ResNeXt3D.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A ResNeXt3D instance.
        """
        ret_config = ResNeXt3D._parse_config(config)
        return cls(**ret_config)
