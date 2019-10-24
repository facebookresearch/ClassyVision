#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from classy_vision.generic.util import is_pos_int, is_pos_int_list

from . import register_model
from .classy_model import ClassyModel, ClassyModelEvaluationMode
from .resnext3d_stage import ResStage
from .resnext3d_stem import ResNeXt3DStem


model_stems = {
    "resnext3d_stem": ResNeXt3DStem,
    # For more types of model stem, add them below
}


@register_model("resnext3d")
class ResNeXt3D(ClassyModel):
    def __init__(
        self,
        input_key,
        input_planes,
        clip_crop_size,
        transformation_type,
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
        zero_init_final_transform_bn,
    ):
        """
            Implementation of 3D ResNe(X)t (https://arxiv.org/pdf/1812.03982.pdf).
            The model consists of one stem, a number of stages, and one or multiple
                heads that are attached to different blocks in the stage.
        Args:
            input_key (str): a key that can index into model input of dict type.
            input_planes (int): the channel dimension of the input. Normally 3 is used
                for rgb input.
            clip_crop_size (int): spatial cropping size of video clip at train time.
            transformation_type (str): the type of residual transformation.
            frames_per_clip (int): No. of frames in a video clip.
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
            width_per_group (int): No. of channels per group in 2nd (group) conv in the
                residual transformation in the first stage
            zero_init_final_transform_bn (bool): if true, the weight of last
                BatchNorm in in the residual transformation is initialized to zero
        """
        super(ResNeXt3D, self).__init__(num_classes=None)

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

        num_stages = len(num_blocks)
        out_planes = [stage_planes * 2 ** i for i in range(num_stages)]
        in_planes = [stem_planes] + out_planes[:-1]
        inner_planes = [
            num_groups * width_per_group * 2 ** i for i in range(num_stages)
        ]

        stages = []
        for s in range(num_stages):
            stage = ResStage(
                s + 2,  # stem is viewed as stage 1, and following stages start from 2
                [in_planes[s]],
                [out_planes[s]],
                [inner_planes[s]],
                [stage_temporal_kernel_basis[s]],
                [temporal_conv_1x1[s]],
                [stage_temporal_stride[s]],
                [stage_spatial_stride[s]],
                [num_blocks[s]],
                [num_groups],
                transformation_type,
                block_callback=self.build_attachable_block,
            )
            stages.append(stage)

        self.stages = nn.Sequential(*stages)
        self._init_parameter(zero_init_final_transform_bn)

    @classmethod
    def from_config(cls, config):
        ret_config = {}
        required_args = [
            "input_planes",
            "clip_crop_size",
            "transformation_type",
            "frames_per_clip",
            "num_blocks",
        ]
        for arg in required_args:
            assert arg in config, "resnext3d model requires argument %s" % arg
            ret_config[arg] = config[arg]
        # Default setting for model stem
        #   stem_planes: No. of output channles of conv op in stem
        #   stem_temporal_kernel: temporal size of conv op in stem
        #   stem_spatial_kernel: spatial size of conv op in stem
        #   stem_maxpool: by default, spatial maxpool op is disabled in stem
        ret_config.update(
            {
                "input_key": config.get("input_key", "video"),
                "stem_name": config.get("stem_name", "resnext3d_stem"),
                "stem_planes": config.get("stem_planes", 64),
                "stem_temporal_kernel": config.get("stem_temporal_kernel", 3),
                "stem_spatial_kernel": config.get("stem_spatial_kernel", 7),
                "stem_maxpool": config.get("stem_maxpool", False),
            }
        )
        # Default setting for model stages 2, 3, 4 and 5
        #   stage_planes: No. of output channel of 1st conv op in stage 2
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
                "zero_init_final_transform_bn": config.get(
                    "zero_init_final_transform_bn", False
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

        return cls(**ret_config)

    def _init_parameter(self, zero_init_final_transform_bn):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) and m.affine:
                if (
                    hasattr(m, "final_transform_bn")
                    and m.final_transform_bn
                    and zero_init_final_transform_bn
                ):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (dict): video input {"video": torch.tensor, "audio": torch.tensor}
        """
        assert type(x) == dict, "input x should be a dictionary (%s)" % str(x)
        assert self._input_key in x, "input key (%s) not in the input" % self._input_key
        out = self.stem([x[self._input_key]])
        out = self.stages(out)

        head_outputs = tuple(self.head_outputs.values())
        if len(head_outputs) == 0:
            raise Exception("Expecting at least one head that generates output")
        elif len(head_outputs) == 1:
            return head_outputs[0]
        else:
            return head_outputs

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
    def output_shape(self):
        return (1, self.num_classes)

    @property
    def model_depth(self):
        return sum(self.num_blocks)

    @property
    def evaluation_mode(self):
        return ClassyModelEvaluationMode.VIDEO_CLIP_AVERAGING

    @property
    def input_key(self):
        return self._input_key

    def validate(self, dataset_output_shape):
        # video model input shape can vary from video to video at testing time.
        # Thus, comparing it with dataset_output_shape will have varying results
        # We skip validation and simply return True
        return True
