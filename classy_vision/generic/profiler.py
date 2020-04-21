#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections.abc as abc
import logging
import operator
from typing import Callable

import torch
import torch.nn as nn
from classy_vision.generic.util import (
    eval_model,
    get_model_dummy_input,
    is_leaf,
    is_on_gpu,
)
from torch.cuda import cudart


def profile(
    model,
    batchsize_per_replica=32,
    input_shape=(3, 224, 224),
    use_nvprof=False,
    input_key=None,
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
    input = get_model_dummy_input(
        model,
        input_shape,
        input_key,
        batchsize=batchsize_per_replica,
        non_blocking=False,
    )
    # perform profiling in eval mode
    with eval_model(model), torch.no_grad():
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


def _get_batchsize_per_replica(x):
    """
    Some layer may take tuple/list/dict/list[dict] as input in forward function. We
    recursively dive into the tuple/list until we meet a tensor and infer the batch size
    """
    while isinstance(x, (list, tuple)):
        assert len(x) > 0, "input x of tuple/list type must have at least one element"
        x = x[0]

    if isinstance(x, (dict,)):
        # index zero is always equal to batch size. select an arbitrary key.
        key_list = list(x.keys())
        x = x[key_list[0]]

    return x.size()[0]


def _layer_flops(layer, x, _):
    """
    Computes the number of FLOPs required for a single layer.

    For common layers, such as Conv1d, the flop compute is implemented in this
    centralized place.
    For other layers, if it defines a method to compute flops with the signature
    below, we will use it to compute flops.

    Class MyModule(nn.Module):
        def flops(self, x):
            ...

    """

    # get layer type:
    typestr = layer.__repr__()
    layer_type = typestr[: typestr.find("(")].strip()
    batchsize_per_replica = _get_batchsize_per_replica(x)
    # 1D convolution:
    if layer_type in ["Conv1d"]:
        # x shape is N x C x W
        out_w = int(
            (x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0])
            / layer.stride[0]
            + 1
        )
        return (
            batchsize_per_replica
            * layer.in_channels
            * layer.out_channels
            * layer.kernel_size[0]
            * out_w
            / layer.groups
        )
    # 2D convolution:
    elif layer_type in ["Conv2d"]:
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
            batchsize_per_replica
            * layer.in_channels
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
            batchsize_per_replica
            * conv.in_channels
            * conv.out_channels
            * conv.kernel_size[0]
            * conv.kernel_size[1]
            * out_h
            * out_w
            / layer.condense_factor
        )
        return count1 + count2

    # non-linearities:
    elif layer_type in ["ReLU", "ReLU6", "Tanh", "Sigmoid", "Softmax"]:
        return x.numel()

    # 2D pooling layers:
    elif layer_type in ["AvgPool2d", "MaxPool2d"]:
        in_h = x.size()[2]
        in_w = x.size()[3]
        if isinstance(layer.kernel_size, int):
            layer.kernel_size = (layer.kernel_size, layer.kernel_size)
        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1]
        out_h = 1 + int(
            (in_h + 2 * layer.padding - layer.kernel_size[0]) / layer.stride
        )
        out_w = 1 + int(
            (in_w + 2 * layer.padding - layer.kernel_size[1]) / layer.stride
        )
        return x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops

    # adaptive avg pool2d
    # This is approximate and works only for downsampling without padding
    # based on aten/src/ATen/native/AdaptiveAveragePooling.cpp
    elif layer_type in ["AdaptiveAvgPool2d"]:
        in_h = x.size()[2]
        in_w = x.size()[3]
        if isinstance(layer.output_size, int):
            out_h, out_w = layer.output_size, layer.output_size
        elif len(layer.output_size) == 1:
            out_h, out_w = layer.output_size[0], layer.output_size[0]
        else:
            out_h, out_w = layer.output_size
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
        bias_ops = layer.bias.numel() if layer.bias is not None else 0
        return x.size()[0] * (weight_ops + bias_ops)

    # batch normalization / layer normalization:
    elif layer_type in [
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "SyncBatchNorm",
        "LayerNorm",
    ]:
        return 2 * x.numel()

    # 3D convolution
    elif layer_type in ["Conv3d"]:
        out_t = int(
            (x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0])
            // layer.stride[0]
            + 1
        )
        out_h = int(
            (x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1])
            // layer.stride[1]
            + 1
        )
        out_w = int(
            (x.size()[4] + 2 * layer.padding[2] - layer.kernel_size[2])
            // layer.stride[2]
            + 1
        )
        return (
            batchsize_per_replica
            * layer.in_channels
            * layer.out_channels
            * layer.kernel_size[0]
            * layer.kernel_size[1]
            * layer.kernel_size[2]
            * out_t
            * out_h
            * out_w
            / layer.groups
        )

    # 3D pooling layers
    elif layer_type in ["AvgPool3d", "MaxPool3d"]:
        in_t = x.size()[2]
        in_h = x.size()[3]
        in_w = x.size()[4]
        if isinstance(layer.kernel_size, int):
            layer.kernel_size = (
                layer.kernel_size,
                layer.kernel_size,
                layer.kernel_size,
            )
        if isinstance(layer.padding, int):
            layer.padding = (layer.padding, layer.padding, layer.padding)
        if isinstance(layer.stride, int):
            layer.stride = (layer.stride, layer.stride, layer.stride)
        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2]
        out_t = 1 + int(
            (in_t + 2 * layer.padding[0] - layer.kernel_size[0]) / layer.stride[0]
        )
        out_h = 1 + int(
            (in_h + 2 * layer.padding[1] - layer.kernel_size[1]) / layer.stride[1]
        )
        out_w = 1 + int(
            (in_w + 2 * layer.padding[2] - layer.kernel_size[2]) / layer.stride[2]
        )
        return batchsize_per_replica * x.size()[1] * out_t * out_h * out_w * kernel_ops

    # adaptive avg pool3d
    # This is approximate and works only for downsampling without padding
    # based on aten/src/ATen/native/AdaptiveAveragePooling3d.cpp
    elif layer_type in ["AdaptiveAvgPool3d"]:
        in_t = x.size()[2]
        in_h = x.size()[3]
        in_w = x.size()[4]
        out_t = layer.output_size[0]
        out_h = layer.output_size[1]
        out_w = layer.output_size[2]
        if out_t > in_t or out_h > in_h or out_w > in_w:
            raise NotImplementedError()
        batchsize_per_replica = x.size()[0]
        num_channels = x.size()[1]
        kt = in_t - out_t + 1
        kh = in_h - out_h + 1
        kw = in_w - out_w + 1
        kernel_ops = kt * kh * kw
        return batchsize_per_replica * num_channels * out_t * out_w * out_h * kernel_ops

    # dropout layer
    elif layer_type in ["Dropout"]:
        # At test time, we do not drop values but scale the feature map by the
        # dropout ratio
        flops = 1
        for dim_size in x.size():
            flops *= dim_size
        return flops

    elif layer_type == "Identity":
        return 0

    elif hasattr(layer, "flops"):
        # If the module already defines a method to compute flops with the signature
        # below, we use it to compute flops
        #
        #   Class MyModule(nn.Module):
        #     def flops(self, x):
        #       ...
        return layer.flops(x)

    # not implemented:
    raise NotImplementedError("FLOPs not implemented for %s layer" % layer_type)


def _layer_activations(layer, x, out):
    """
    Computes the number of activations produced by a single layer.

    Activations are counted only for convolutional layers. To override this behavior, a
    layer can define a method to compute activations with the signature below, which
    will be used to compute the activations instead.

    Class MyModule(nn.Module):
        def activations(self, x, out):
            ...
    """
    if hasattr(layer, "activations"):
        return layer.activations(x, out)
    return out.numel() if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) else 0


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


class _ComplexityComputer:
    def __init__(self, compute_fn: Callable, count_unique: bool):
        self.compute_fn = compute_fn
        self.count_unique = count_unique
        self.count = 0
        self.seen_modules = set()

    def compute(self, layer, x, out, module_name):
        if self.count_unique and module_name in self.seen_modules:
            return
        self.count += self.compute_fn(layer, x, out)
        self.seen_modules.add(module_name)


def _patched_computation_module(module, complexity_computer, module_name):
    """
    Patch the module to compute a module's parameters, like FLOPs.

    Calls compute_fn and passes the results to the complexity computer.
    """
    ty = type(module)
    typestring = module.__repr__()

    class ComputeModule(ty):
        orig_type = ty

        def _original_forward(self, *args, **kwargs):
            return ty.forward(self, *args, **kwargs)

        def forward(self, *args, **kwargs):
            out = self._original_forward(*args, **kwargs)
            complexity_computer.compute(self, args[0], out, module_name)
            return out

        def __repr__(self):
            return typestring

    return ComputeModule


def modify_forward(model, complexity_computer, prefix="", patch_attr=None):
    """
    Modify forward pass to measure a module's parameters, like FLOPs.
    """
    if is_leaf(model) or (patch_attr is not None and hasattr(model, patch_attr)):
        model.__class__ = _patched_computation_module(
            model, complexity_computer, prefix
        )

    else:
        for name, child in model.named_children():
            modify_forward(
                child,
                complexity_computer,
                prefix=f"{prefix}.{name}",
                patch_attr=patch_attr,
            )

    return model


def restore_forward(model, patch_attr=None):
    """
    Restore original forward in model.
    """
    if is_leaf(model) or (patch_attr is not None and hasattr(model, patch_attr)):
        model.__class__ = model.orig_type

    else:
        for child in model.children():
            restore_forward(child, patch_attr=patch_attr)

    return model


def compute_complexity(
    model,
    compute_fn,
    input_shape,
    input_key=None,
    patch_attr=None,
    compute_unique=False,
):
    """
    Compute the complexity of a forward pass.

    Args:
        compute_unique: If True, the compexity for a given module is only calculated
            once. Otherwise, it is counted every time the module is called.

    TODO(@mannatsingh): We have some assumptions about only modules which are leaves
        or have patch_attr defined. This should be fixed and generalized if possible.
    """
    # assertions, input, and upvalue in which we will perform the count:
    assert isinstance(model, nn.Module)

    if not isinstance(input_shape, abc.Sequence) and not isinstance(input_shape, dict):
        return None
    else:
        input = get_model_dummy_input(model, input_shape, input_key)

    complexity_computer = _ComplexityComputer(compute_fn, compute_unique)

    # measure FLOPs:
    modify_forward(model, complexity_computer, patch_attr=patch_attr)
    try:
        # compute complexity in eval mode
        with eval_model(model), torch.no_grad():
            model.forward(input)
    except NotImplementedError as err:
        raise err
    finally:
        restore_forward(model, patch_attr=patch_attr)

    return complexity_computer.count


def compute_flops(model, input_shape=(3, 224, 224), input_key=None):
    """
    Compute the number of FLOPs needed for a forward pass.
    """
    return compute_complexity(
        model, _layer_flops, input_shape, input_key, patch_attr="flops"
    )


def compute_activations(model, input_shape=(3, 224, 224), input_key=None):
    """
    Compute the number of activations created in a forward pass.
    """
    return compute_complexity(
        model, _layer_activations, input_shape, input_key, patch_attr="activations"
    )


def count_params(model):
    """
    Count the number of parameters in a model.
    """
    assert isinstance(model, nn.Module)
    return sum((parameter.nelement() for parameter in model.parameters()))
