#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from .r2plus1_util import r2plus1_unit


class BasicTransformation(nn.Module):
    """
    Basic transformation: 3x3x3 group conv, 3x3x3 group conv
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temporal_stride,
        spatial_stride,
        groups,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        **kwargs
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            groups (int): number of groups for the convolution.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(BasicTransformation, self).__init__()
        self._construct_model(
            dim_in,
            dim_out,
            temporal_stride,
            spatial_stride,
            groups,
            inplace_relu,
            bn_eps,
            bn_mmt,
        )

    def _construct_model(
        self,
        dim_in,
        dim_out,
        temporal_stride,
        spatial_stride,
        groups,
        inplace_relu,
        bn_eps,
        bn_mmt,
    ):
        # 3x3x3 group conv, BN, ReLU.
        branch2a = nn.Conv3d(
            dim_in,
            dim_out,
            [3, 3, 3],  # kernel
            stride=[temporal_stride, spatial_stride, spatial_stride],
            padding=[1, 1, 1],
            groups=groups,
            bias=False,
        )
        branch2a_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2a_relu = nn.ReLU(inplace=inplace_relu)
        # 3x3x3 group conv, BN, ReLU.
        branch2b = nn.Conv3d(
            dim_out,
            dim_out,
            [3, 3, 3],  # kernel
            stride=[1, 1, 1],
            padding=[1, 1, 1],
            groups=groups,
            bias=False,
        )
        branch2b_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2b_bn.final_transform_op = True

        self.transform = nn.Sequential(
            branch2a, branch2a_bn, branch2a_relu, branch2b, branch2b_bn
        )

    def forward(self, x):
        return self.transform(x)


class BasicR2Plus1DTransformation(BasicTransformation):
    """
    Basic transformation: 3x3x3 group conv, 3x3x3 group conv
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temporal_stride,
        spatial_stride,
        groups,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        **kwargs
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            groups (int): number of groups for the convolution.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(BasicR2Plus1DTransformation, self).__init__(
            dim_in,
            dim_out,
            temporal_stride,
            spatial_stride,
            groups,
            inplace_relu=inplace_relu,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )

    def _construct_model(
        self,
        dim_in,
        dim_out,
        temporal_stride,
        spatial_stride,
        groups,
        inplace_relu,
        bn_eps,
        bn_mmt,
    ):
        # Implementation of R(2+1)D operation <https://arxiv.org/abs/1711.11248>.
        # decompose the original 3D conv into one 2D spatial conv and one
        # 1D temporal conv
        branch2a = r2plus1_unit(
            dim_in,
            dim_out,
            temporal_stride,
            spatial_stride,
            groups,
            inplace_relu,
            bn_eps,
            bn_mmt,
        )
        branch2a_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2a_relu = nn.ReLU(inplace=inplace_relu)

        branch2b = r2plus1_unit(
            dim_out,
            dim_out,
            1,  # temporal_stride
            1,  # spatial_stride
            groups,
            inplace_relu,
            bn_eps,
            bn_mmt,
        )
        branch2b_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2b_bn.final_transform_op = True

        self.transform = nn.Sequential(
            branch2a, branch2a_bn, branch2a_relu, branch2b, branch2b_bn
        )


class PostactivatedBottleneckTransformation(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temporal_stride,
        spatial_stride,
        num_groups,
        dim_inner,
        temporal_kernel_size=3,
        temporal_conv_1x1=True,
        spatial_stride_1x1=False,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        **kwargs
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temporal_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            temporal_conv_1x1 (bool): if True, do temporal convolution in the fist
                1x1 Conv3d. Otherwise, do it in the second 3x3 Conv3d
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            num_groups (int): number of groups for the convolution.
            dim_inner (int): the inner dimension of the block.
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            spatial_stride_1x1 (bool): if True, apply spatial_stride to 1x1 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(PostactivatedBottleneckTransformation, self).__init__()
        (temporal_kernel_size_1x1, temporal_kernel_size_3x3) = (
            (temporal_kernel_size, 1)
            if temporal_conv_1x1
            else (1, temporal_kernel_size)
        )
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3.
        (str1x1, str3x3) = (
            (spatial_stride, 1) if spatial_stride_1x1 else (1, spatial_stride)
        )
        # Tx1x1 conv, BN, ReLU.
        self.branch2a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[temporal_kernel_size_1x1, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[temporal_kernel_size_1x1 // 2, 0, 0],
            bias=False,
        )
        self.branch2a_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2a_relu = nn.ReLU(inplace=inplace_relu)
        # Tx3x3 group conv, BN, ReLU.
        self.branch2b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [temporal_kernel_size_3x3, 3, 3],
            stride=[temporal_stride, str3x3, str3x3],
            padding=[temporal_kernel_size_3x3 // 2, 1, 1],
            groups=num_groups,
            bias=False,
        )
        self.branch2b_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2b_relu = nn.ReLU(inplace=inplace_relu)
        # 1x1x1 conv, BN.
        self.branch2c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.branch2c_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        self.branch2c_bn.final_transform_op = True

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.branch2a(x)
        x = self.branch2a_bn(x)
        x = self.branch2a_relu(x)

        # Branch2b.
        x = self.branch2b(x)
        x = self.branch2b_bn(x)
        x = self.branch2b_relu(x)

        # Branch2c
        x = self.branch2c(x)
        x = self.branch2c_bn(x)
        return x


class PreactivatedBottleneckTransformation(nn.Module):
    """
    Bottleneck transformation with pre-activation, which includes BatchNorm3D
        and ReLu. Conv3D kernsl are Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel (https://arxiv.org/abs/1603.05027).
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temporal_stride,
        spatial_stride,
        num_groups,
        dim_inner,
        temporal_kernel_size=3,
        temporal_conv_1x1=True,
        spatial_stride_1x1=False,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        disable_pre_activation=False,
        **kwargs
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temporal_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            temporal_conv_1x1 (bool): if True, do temporal convolution in the fist
                1x1 Conv3d. Otherwise, do it in the second 3x3 Conv3d
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            num_groups (int): number of groups for the convolution.
            dim_inner (int): the inner dimension of the block.
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            spatial_stride_1x1 (bool): if True, apply spatial_stride to 1x1 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            disable_pre_activation (bool): If true, disable pre activation,
                including BatchNorm3D and ReLU.
        """
        super(PreactivatedBottleneckTransformation, self).__init__()
        (temporal_kernel_size_1x1, temporal_kernel_size_3x3) = (
            (temporal_kernel_size, 1)
            if temporal_conv_1x1
            else (1, temporal_kernel_size)
        )
        (str1x1, str3x3) = (
            (spatial_stride, 1) if spatial_stride_1x1 else (1, spatial_stride)
        )

        self.disable_pre_activation = disable_pre_activation
        if not disable_pre_activation:
            self.branch2a_bn = nn.BatchNorm3d(dim_in, eps=bn_eps, momentum=bn_mmt)
            self.branch2a_relu = nn.ReLU(inplace=inplace_relu)

        self.branch2a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[temporal_kernel_size_1x1, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[temporal_kernel_size_1x1 // 2, 0, 0],
            bias=False,
        )
        # Tx3x3 group conv, BN, ReLU.
        self.branch2b_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2b_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [temporal_kernel_size_3x3, 3, 3],
            stride=[temporal_stride, str3x3, str3x3],
            padding=[temporal_kernel_size_3x3 // 2, 1, 1],
            groups=num_groups,
            bias=False,
        )
        # 1x1x1 conv, BN.
        self.branch2c_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2c_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.branch2c.final_transform_op = True

    def forward(self, x):
        # Branch2a
        if not self.disable_pre_activation:
            x = self.branch2a_bn(x)
            x = self.branch2a_relu(x)
        x = self.branch2a(x)
        # Branch2b
        x = self.branch2b_bn(x)
        x = self.branch2b_relu(x)
        x = self.branch2b(x)
        # Branch2c
        x = self.branch2c_bn(x)
        x = self.branch2c_relu(x)
        x = self.branch2c(x)
        return x


residual_transformations = {
    "basic_r2plus1d_transformation": BasicR2Plus1DTransformation,
    "basic_transformation": BasicTransformation,
    "postactivated_bottleneck_transformation": PostactivatedBottleneckTransformation,
    "preactivated_bottleneck_transformation": PreactivatedBottleneckTransformation,
    # For more types of residual transformations, add them below
}


class PostactivatedShortcutTransformation(nn.Module):
    """
    Skip connection used in ResNet3D model.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temporal_stride,
        spatial_stride,
        bn_eps=1e-5,
        bn_mmt=0.1,
        **kwargs
    ):
        super(PostactivatedShortcutTransformation, self).__init__()
        # Use skip connection with projection if dim or spatial/temporal res change.
        assert (dim_in != dim_out) or (spatial_stride != 1) or (temporal_stride != 1)
        self.branch1 = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=1,
            stride=[temporal_stride, spatial_stride, spatial_stride],
            padding=0,
            bias=False,
        )
        self.branch1_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)

    def forward(self, x):
        return self.branch1_bn(self.branch1(x))


class PreactivatedShortcutTransformation(nn.Module):
    """
    Skip connection with pre-activation, which includes BatchNorm3D and ReLU,
        in ResNet3D model (https://arxiv.org/abs/1603.05027).
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temporal_stride,
        spatial_stride,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        disable_pre_activation=False,
        **kwargs
    ):
        super(PreactivatedShortcutTransformation, self).__init__()
        # Use skip connection with projection if dim or spatial/temporal res change.
        assert (dim_in != dim_out) or (spatial_stride != 1) or (temporal_stride != 1)
        if not disable_pre_activation:
            self.branch1_bn = nn.BatchNorm3d(dim_in, eps=bn_eps, momentum=bn_mmt)
            self.branch1_relu = nn.ReLU(inplace=inplace_relu)
        self.branch1 = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=1,
            stride=[temporal_stride, spatial_stride, spatial_stride],
            padding=0,
            bias=False,
        )

    def forward(self, x):
        if hasattr(self, "branch1_bn") and hasattr(self, "branch1_relu"):
            x = self.branch1_relu(self.branch1_bn(x))
        x = self.branch1(x)
        return x


skip_transformations = {
    "postactivated_shortcut": PostactivatedShortcutTransformation,
    "preactivated_shortcut": PreactivatedShortcutTransformation,
    # For more types of skip transformations, add them below
}


class ResBlock(nn.Module):
    """
    Residual block with skip connection.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        dim_inner,
        temporal_kernel_size,
        temporal_conv_1x1,
        temporal_stride,
        spatial_stride,
        skip_transformation_type,
        residual_transformation_type,
        num_groups=1,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        disable_pre_activation=False,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            dim_inner (int): the inner dimension of the block.
            temporal_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            temporal_conv_1x1 (bool): Only useful for PostactivatedBottleneckTransformation.
                if True, do temporal convolution in the fist 1x1 Conv3d.
                Otherwise, do it in the second 3x3 Conv3d
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            stride (int): the stride of the bottleneck.
            skip_transformation_type (str): the type of skip transformation
            residual_transformation_type (str): the type of residual transformation
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            disable_pre_activation (bool): If true, disable the preactivation,
                which includes BatchNorm3D and ReLU.
        """
        super(ResBlock, self).__init__()

        assert skip_transformation_type in skip_transformations, (
            "unknown skip transformation: %s" % skip_transformation_type
        )

        if (dim_in != dim_out) or (spatial_stride != 1) or (temporal_stride != 1):
            self.skip = skip_transformations[skip_transformation_type](
                dim_in,
                dim_out,
                temporal_stride,
                spatial_stride,
                bn_eps=bn_eps,
                bn_mmt=bn_mmt,
                disable_pre_activation=disable_pre_activation,
            )

        assert residual_transformation_type in residual_transformations, (
            "unknown residual transformation: %s" % residual_transformation_type
        )
        self.residual = residual_transformations[residual_transformation_type](
            dim_in,
            dim_out,
            temporal_stride,
            spatial_stride,
            num_groups,
            dim_inner,
            temporal_kernel_size=temporal_kernel_size,
            temporal_conv_1x1=temporal_conv_1x1,
            disable_pre_activation=disable_pre_activation,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        if hasattr(self, "skip"):
            x = self.skip(x) + self.residual(x)
        else:
            x = x + self.residual(x)
        x = self.relu(x)
        return x
