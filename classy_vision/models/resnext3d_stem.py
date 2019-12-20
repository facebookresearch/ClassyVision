#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from .r2plus1_util import r2plus1_unit


class ResNeXt3DStemSinglePathway(nn.Module):
    """
    ResNe(X)t 3D basic stem module. Assume a single pathway.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        maxpool=True,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            maxpool (bool): If true, perform max pooling.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(ResNeXt3DStemSinglePathway, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.bn_eps = bn_eps
        self.bn_mmt = bn_mmt
        self.maxpool = maxpool

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(dim_out, eps=self.bn_eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        if self.maxpool:
            self.pool_layer = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.pool_layer(x)
        return x


class R2Plus1DStemSinglePathway(ResNeXt3DStemSinglePathway):
    """
    R(2+1)D basic stem module. Assume a single pathway.
    Performs spatial convolution, temporal convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        maxpool=True,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            maxpool (bool): If true, perform max pooling.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(R2Plus1DStemSinglePathway, self).__init__(
            dim_in,
            dim_out,
            kernel,
            stride,
            padding,
            maxpool=maxpool,
            inplace_relu=inplace_relu,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )

    def _construct_stem(self, dim_in, dim_out):

        assert (
            self.stride[1] == self.stride[2]
        ), "Only support identical height stride and width stride"
        self.conv = r2plus1_unit(
            dim_in,
            dim_out,
            self.stride[0],  # temporal_stride
            self.stride[1],  # spatial_stride
            1,  # groups
            self.inplace_relu,
            self.bn_eps,
            self.bn_mmt,
            dim_mid=45,  # hard-coded middle channels
        )
        self.bn = nn.BatchNorm3d(dim_out, eps=self.bn_eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        if self.maxpool:
            self.pool_layer = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            )


class ResNeXt3DStemMultiPathway(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        maxpool=(True,),
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, SlowOnly
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            maxpool (iterable): At training time, when crop size is 224 x 224, do max
                pooling. When crop size is 112 x 112, skip max pooling.
                Default value is a (True,)
        """
        super(ResNeXt3DStemMultiPathway, self).__init__()

        assert (
            len({len(dim_in), len(dim_out), len(kernel), len(stride), len(padding)})
            == 1
        ), "Input pathway dimensions are not consistent."
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.bn_eps = bn_eps
        self.bn_mmt = bn_mmt
        self.maxpool = maxpool

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        assert type(dim_in) == list
        assert all(dim > 0 for dim in dim_in)
        assert type(dim_out) == list
        assert all(dim > 0 for dim in dim_out)

        self.blocks = {}
        for p in range(len(dim_in)):
            stem = ResNeXt3DStemSinglePathway(
                dim_in[p],
                dim_out[p],
                self.kernel[p],
                self.stride[p],
                self.padding[p],
                inplace_relu=self.inplace_relu,
                bn_eps=self.bn_eps,
                bn_mmt=self.bn_mmt,
                maxpool=self.maxpool[p],
            )
            stem_name = self._stem_name(p)
            self.add_module(stem_name, stem)
            self.blocks[stem_name] = stem

    def _stem_name(self, path_idx):
        return "stem-path{}".format(path_idx)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for p in range(len(x)):
            stem_name = self._stem_name(p)
            x[p] = self.blocks[stem_name](x[p])
        return x


class R2Plus1DStemMultiPathway(ResNeXt3DStemMultiPathway):
    """
    Video R(2+1)D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        maxpool=(True,),
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, SlowOnly
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            maxpool (iterable): At training time, when crop size is 224 x 224, do max
                pooling. When crop size is 112 x 112, skip max pooling.
                Default value is a (True,)
        """
        super(R2Plus1DStemMultiPathway, self).__init__(
            dim_in,
            dim_out,
            kernel,
            stride,
            padding,
            inplace_relu=inplace_relu,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
            maxpool=maxpool,
        )

    def _construct_stem(self, dim_in, dim_out):
        assert type(dim_in) == list
        assert all(dim > 0 for dim in dim_in)
        assert type(dim_out) == list
        assert all(dim > 0 for dim in dim_out)

        self.blocks = {}
        for p in range(len(dim_in)):
            stem = R2Plus1DStemSinglePathway(
                dim_in[p],
                dim_out[p],
                self.kernel[p],
                self.stride[p],
                self.padding[p],
                inplace_relu=self.inplace_relu,
                bn_eps=self.bn_eps,
                bn_mmt=self.bn_mmt,
                maxpool=self.maxpool[p],
            )
            stem_name = self._stem_name(p)
            self.add_module(stem_name, stem)
            self.blocks[stem_name] = stem


class ResNeXt3DStem(nn.Module):
    def __init__(
        self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
    ):
        super(ResNeXt3DStem, self).__init__()
        self._construct_stem(
            temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
        )

    def _construct_stem(
        self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
    ):
        self.stem = ResNeXt3DStemMultiPathway(
            [input_planes],
            [stem_planes],
            [[temporal_kernel, spatial_kernel, spatial_kernel]],
            [[1, 2, 2]],  # stride
            [
                [temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2]
            ],  # padding
            maxpool=[maxpool],
        )

    def forward(self, x):
        return self.stem(x)


class R2Plus1DStem(ResNeXt3DStem):
    def __init__(
        self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
    ):
        super(R2Plus1DStem, self).__init__(
            temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
        )

    def _construct_stem(
        self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
    ):
        self.stem = R2Plus1DStemMultiPathway(
            [input_planes],
            [stem_planes],
            [[temporal_kernel, spatial_kernel, spatial_kernel]],
            [[1, 2, 2]],  # stride
            [
                [temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2]
            ],  # padding
            maxpool=[maxpool],
        )
