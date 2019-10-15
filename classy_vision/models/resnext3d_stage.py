#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from .resnext3d_block import ResBlock


class ResStage(nn.Module):
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
        transformation_type,
        block_callback=None,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
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
            transformation_type (str): the type of residual transformation
        """
        super(ResStage, self).__init__()

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

        temporal_kernel_sizes = [
            (temporal_kernel_basis[i] * num_blocks[i])[: num_blocks[i]]
            for i in range(len(temporal_kernel_basis))
        ]

        self.blocks = nn.ModuleDict()
        for p in range(self.num_pathways):
            for i in range(self.num_blocks[p]):
                # Retrieve the transformation function.
                # Construct the block.
                res_block = ResBlock(
                    dim_in[p] if i == 0 else dim_out[p],
                    dim_out[p],
                    dim_inner[p],
                    temporal_kernel_sizes[p][i],
                    temporal_conv_1x1[p],
                    temporal_stride[p] if i == 0 else 1,
                    spatial_stride[p] if i == 0 else 1,
                    transformation_type,
                    num_groups=num_groups[p],
                    inplace_relu=inplace_relu,
                    bn_eps=bn_eps,
                    bn_mmt=bn_mmt,
                )
                block_name = self._block_name(stage_idx, p, i)
                if block_callback:
                    res_block = block_callback(block_name, res_block)
                self.blocks[block_name] = res_block

    def _block_name(self, stage_idx, path_idx, block_idx):
        # offset path_idx by 1 and block_idx by 1 to conform to convention
        return "pathway{}-stage{}-block{}".format(
            path_idx + 1, stage_idx, block_idx + 1
        )

    def forward(self, inputs):
        output = []
        for p in range(self.num_pathways):
            x = inputs[p]
            for i in range(self.num_blocks[p]):
                block_name = self._block_name(self.stage_idx, p, i)
                x = self.blocks[block_name](x)
            output.append(x)
        return output
