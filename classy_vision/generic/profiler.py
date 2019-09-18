#!/usr/bin/env python3

import collections.abc as abc
import logging
import operator

import torch
import torch.nn as nn
from classy_vision.generic.util import is_leaf, is_on_gpu
from torch.cuda import cudart


def profile(
    model, batchsize_per_replica=32, input_shape=(3, 224, 224), use_nvprof=False
):
    """
    Performs CPU or GPU profiling of the specified model on the specified input.
    """

    # assertions:
    if use_nvprof:
        raise NotImplementedError
        # FIXME (mannatsingh): in case of use_nvprof, exit() is called at the end
        # and we do not return a profile.
        assert is_on_gpu(model), "can only nvprof model that lives on GPU"
        logging.info("CUDA profiling: Make sure you are running under nvprof!")

    # input for model:
    input = torch.zeros(batchsize_per_replica, *input_shape)
    if is_on_gpu(model):
        input = input.cuda(non_blocking=False)

    # perform profiling:
    with torch.no_grad():
        model(input)  # warm up CUDA memory allocator and profiler
        if use_nvprof:  # nvprof profiling (TODO: Can we infer this?)
            cudart().cudaProfilerStart()
            model(input)
            cudart().cudaProfilerStop()
            exit()  # exit gracefully
        else:  # regular profiling
            with torch.autograd.profiler.profile(use_cuda=True) as profiler:
                model(input)
                return profiler


def _layer_flops(layer, x):
    """
    Computes the number of FLOPs required for a single layer.
    """

    # get layer type:
    typestr = layer.__repr__()
    layer_type = typestr[: typestr.find("(")].strip()

    # 2D convolution:
    if layer_type in ["Conv2d"]:
        out_h = int(
            (x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0])
            / layer.stride[0]
            + 1
        )
        out_w = int(
            (x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1])
            / layer.stride[1]
            + 1
        )
        return (
            layer.in_channels
            * layer.out_channels
            * layer.kernel_size[0]
            * layer.kernel_size[1]
            * out_h
            * out_w
            / layer.groups
        )

    # learned group convolution:
    elif layer_type in ["LearnedGroupConv"]:
        conv = layer.conv
        out_h = int(
            (x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) / conv.stride[0]
            + 1
        )
        out_w = int(
            (x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) / conv.stride[1]
            + 1
        )
        count1 = _layer_flops(layer.relu, x) + _layer_flops(layer.norm, x)
        count2 = (
            conv.in_channels
            * conv.out_channels
            * conv.kernel_size[0]
            * conv.kernel_size[1]
            * out_h
            * out_w
            / layer.condense_factor
        )
        return count1 + count2

    # non-linearities:
    elif layer_type in ["ReLU", "Tanh", "Sigmoid"]:
        return x.numel()

    # 2D pooling layers:
    elif layer_type in ["AvgPool2d", "MaxPool2d"]:
        in_w = x.size()[2]
        if isinstance(layer.kernel_size, int):
            layer.kernel_size = (layer.kernel_size, layer.kernel_size)
        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1]
        out_w = 1 + int(
            (in_w + 2 * layer.padding - layer.kernel_size[1]) / layer.stride
        )
        out_h = 1 + int(
            (in_w + 2 * layer.padding - layer.kernel_size[0]) / layer.stride
        )
        return x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops

    # adaptive avg pool2d
    # This is approximate and works only for downsampling without padding
    # based on aten/src/ATen/native/AdaptiveAveragePooling.cpp
    elif layer_type in ["AdaptiveAvgPool2d"]:
        in_h = x.size()[2]
        in_w = x.size()[3]
        out_h = layer.output_size[0]
        out_w = layer.output_size[1]
        if out_h > in_h or out_w > in_w:
            raise NotImplementedError()
        batchsize_per_replica = x.size()[0]
        num_channels = x.size()[1]
        kh = in_h - out_h + 1
        kw = in_w - out_w + 1
        kernel_ops = kh * kw
        return batchsize_per_replica * num_channels * out_h * out_w * kernel_ops

    # linear layer:
    elif layer_type in ["Linear"]:
        weight_ops = layer.weight.numel()
        bias_ops = layer.bias.numel()
        return x.size()[0] * (weight_ops + bias_ops)

    # batch normalization:
    elif layer_type in ["BatchNorm2d"]:
        return 2 * x.size()[0]

    # not implemented:
    raise NotImplementedError("FLOPs not implemented for %s layer" % layer_type)


def summarize_profiler_info(prof):
    """
    Summarizes the statistics in the specified profiler.
    """

    # create sorted list of times per operator:
    op2time = {}
    for item in prof.key_averages():
        op2time[item.key] = (
            item.cpu_time_total / 1000.0,
            item.cuda_time_total / 1000.0,
        )  # to milliseconds
    op2time = sorted(op2time.items(), key=operator.itemgetter(1), reverse=True)

    # created string containing information:
    str = "\n%s\tCPU Time\tCUDA Time\n" % ("Key".rjust(20))
    for (key, value) in op2time:
        str += "%s\t%2.5f ms\t%2.5f ms\n" % (key.rjust(20), value[0], value[1])
    return str


def _flops_module(module, flops_list):
    """
    Convert module into FLOP-counting module.
    """
    ty = type(module)
    typestring = module.__repr__()

    class FLOPsModule(ty):
        orig_type = ty

        def _original_forward(self, *args, **kwargs):
            return ty.forward(self, *args, **kwargs)

        def forward(self, *args, **kwargs):
            flops_list.append(_layer_flops(self, args[0]))
            return self._original_forward(*args, **kwargs)

        def __repr__(self):
            return typestring

    return FLOPsModule


def modify_forward(model, flops_list):
    """
    Modify forward pass to measure FLOPs:
    """
    if is_leaf(model):
        model.__class__ = _flops_module(model, flops_list)
    for child in model.children():
        modify_forward(child, flops_list)

    return model


def restore_forward(model):
    """
    Restore original forward in model:
    """
    if is_leaf(model):
        model.__class__ = model.orig_type
    for child in model.children():
        restore_forward(child)

    return model


def compute_flops(model, input_shape=(3, 244, 244)):
    """
    Compute the number of FLOPs needed for a forward pass.
    """

    # assertions, input, and upvalue in which we will perform the count:
    assert isinstance(model, nn.Module)
    if not isinstance(input_shape, abc.Sequence):
        return None
    shape = (1,) + tuple(input_shape)
    input = torch.zeros(shape)
    if next(model.parameters()).is_cuda:
        input = input.cuda()

    flops_list = []

    # measure FLOPs:
    modify_forward(model, flops_list)
    try:
        model.forward(input)
    except NotImplementedError as err:
        raise err
    finally:
        restore_forward(model)
    return sum(flops_list)


def count_params(model):
    """
    Count the number of parameters in a model.
    """
    assert isinstance(model, nn.Module)
    count = 0
    for child in model.children():
        if is_leaf(child):
            if hasattr(child, "_mask"):  # for masked modules (like LGC)
                count += child._mask.long().sum().item()
                # FIXME: BatchNorm parameters in LGC are not counted.
            else:  # for regular modules
                for p in child.parameters():
                    count += p.nelement()
        else:
            count += count_params(child)
    return count
