#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch.nn as nn

from .resnext3d_block import ResBlock


class ResStageBase(nn.Module):
    def __init__(
        self,
        stage_idx,
        dim_in,
        dim_out,
        dim_inner,
        temporal_kernel_basis,
        temporal_conv_1x1,
        temporal_stride,
        spatial_stride,
        num_blocks,
        num_groups,
    ):
        super(ResStageBase, self).__init__()

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temporal_kernel_basis),
                    len(temporal_conv_1x1),
                    len(temporal_stride),
                    len(spatial_stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                }
            )
            == 1
        )

        self.stage_idx = stage_idx
        self.num_blocks = num_blocks
        self.num_pathways = len(self.num_blocks)

        self.temporal_kernel_sizes = [
            (temporal_kernel_basis[i] * num_blocks[i])[: num_blocks[i]]
            for i in range(len(temporal_kernel_basis))
        ]

    def _block_name(self, pathway_idx, stage_idx, block_idx):
        return "pathway{}-stage{}-block{}".format(pathway_idx, stage_idx, block_idx)

    def _pathway_name(self, pathway_idx):
        return "pathway{}".format(pathway_idx)

    def forward(self, inputs):
        output = []
        for p in range(self.num_pathways):
            x = inputs[p]
            pathway_module = getattr(self, self._pathway_name(p))
            output.append(pathway_module(x))
        return output


class ResStage(ResStageBase):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, SlowOnly), and multi-pathway (SlowFast) cases.
        More details can be found here:
        "Slowfast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        stage_idx,
        dim_in,
        dim_out,
        dim_inner,
        temporal_kernel_basis,
        temporal_conv_1x1,
        temporal_stride,
        spatial_stride,
        num_blocks,
        num_groups,
        skip_transformation_type,
        residual_transformation_type,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        disable_pre_activation=False,
        final_stage=False,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            stage_idx (int): integer index of stage.
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_inner (list): list of the p inner channel dimensions of the
                input.
                Different channel dimensions control the input dimension of
                different pathways.
            temporal_kernel_basis (list): Basis of temporal kernel sizes for each of
                the stage.
            temporal_conv_1x1 (list): Only useful for BottleneckBlock.
                In a pathaway, if True, do temporal convolution in the fist 1x1 Conv3d.
                Otherwise, do it in the second 3x3 Conv3d
            temporal_stride (list): the temporal stride of the bottleneck.
            spatial_stride (list): the spatial_stride of the bottleneck.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            skip_transformation_type (str): the type of skip transformation
            residual_transformation_type (str): the type of residual transformation
            disable_pre_activation (bool): If true, disable the preactivation,
                which includes BatchNorm3D and ReLU.
            final_stage (bool): If true, this is the last stage in the model.
        """
        super(ResStage, self).__init__(
            stage_idx,
            dim_in,
            dim_out,
            dim_inner,
            temporal_kernel_basis,
            temporal_conv_1x1,
            temporal_stride,
            spatial_stride,
            num_blocks,
            num_groups,
        )

        for p in range(self.num_pathways):
            blocks = []
            for i in range(self.num_blocks[p]):
                # Retrieve the transformation function.
                # Construct the block.
                block_disable_pre_activation = (
                    True if disable_pre_activation and i == 0 else False
                )
                res_block = ResBlock(
                    dim_in[p] if i == 0 else dim_out[p],
                    dim_out[p],
                    dim_inner[p],
                    self.temporal_kernel_sizes[p][i],
                    temporal_conv_1x1[p],
                    temporal_stride[p] if i == 0 else 1,
                    spatial_stride[p] if i == 0 else 1,
                    skip_transformation_type,
                    residual_transformation_type,
                    num_groups=num_groups[p],
                    inplace_relu=inplace_relu,
                    bn_eps=bn_eps,
                    bn_mmt=bn_mmt,
                    disable_pre_activation=block_disable_pre_activation,
                )
                block_name = self._block_name(p, stage_idx, i)
                blocks.append((block_name, res_block))

            if final_stage and (
                residual_transformation_type == "preactivated_bottleneck_transformation"
            ):
                # For pre-activation residual transformation, we conduct
                # activation in the final stage before continuing forward pass
                # through the head
                activate_bn = nn.BatchNorm3d(dim_out[p])
                activate_relu = nn.ReLU(inplace=True)
                activate_bn_name = "-".join([block_name, "bn"])
                activate_relu_name = "-".join([block_name, "relu"])
                blocks.append((activate_bn_name, activate_bn))
                blocks.append((activate_relu_name, activate_relu))

            self.add_module(self._pathway_name(p), nn.Sequential(OrderedDict(blocks)))
