#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections.abc as abc
import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from classy_vision.generic.util import (
    eval_model,
    get_model_dummy_input,
    is_leaf,
    is_on_gpu,
)
from torch.cuda import cudart


class ClassyProfilerError(Exception):
    pass


class ClassyProfilerNotImplementedError(ClassyProfilerError):
    def __init__(self, module: nn.Module):
        self.module = module
        super().__init__(f"Profiling not implemented for module: {self.module}")


def profile(
    model: nn.Module,
    batchsize_per_replica: int = 32,
    input_shape: Tuple[int] = (3, 224, 224),
    use_nvprof: bool = False,
    input_key: Optional[Union[str, List[str]]] = None,
):
    """
    Performs CPU or GPU profiling of the specified model on the specified input.
    """
    # assertions:
    if use_nvprof:
        raise ClassyProfilerError("Profiling not supported with nvprof")
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


def get_shape(x: Union[Tuple, List, Dict]) -> Union[Tuple, List, Dict]:
    """
    Some layer may take/generate tuple/list/dict/list[dict] as input/output in forward function.
    We recursively query tensor shape.
    """
    if isinstance(x, (list, tuple)):
        assert len(x) > 0, "x of tuple/list type must have at least one element"
        return [get_shape(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: get_shape(v) for k, v in x.items()}
    else:
        assert isinstance(x, torch.Tensor), "x is expected to be a torch tensor"
        return x.size()


def _get_batchsize_per_replica(x: Union[Tuple, List, Dict]) -> int:
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


def _layer_flops(layer: nn.Module, x: Any, y: Any, verbose: bool = False) -> int:
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

    flops = None
    # 1D convolution:
    if layer_type in ["Conv1d"]:
        # x shape is N x C x W
        out_w = int(
            (x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0])
            / layer.stride[0]
            + 1
        )
        flops = (
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
        flops = (
            batchsize_per_replica
            * layer.in_channels
            * layer.out_channels
            * layer.kernel_size[0]
            * layer.kernel_size[1]
            * out_h
            * out_w
            / layer.groups
        )

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
        flops = (
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
        flops = count1 + count2

    # non-linearities are not considered in MAC counting
    elif layer_type in ["ReLU", "ReLU6", "Tanh", "Sigmoid", "Softmax"]:
        flops = 0

    elif layer_type in [
        "MaxPool1d",
        "MaxPool2d",
        "MaxPool3d",
        "AdaptiveMaxPool1d",
        "AdaptiveMaxPool2d",
        "AdaptiveMaxPool3d",
    ]:
        flops = 0

    elif layer_type in ["AvgPool1d", "AvgPool2d", "AvgPool3d"]:
        kernel_ops = 1
        flops = kernel_ops * y.numel()

    elif layer_type in ["AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d"]:
        assert isinstance(layer.output_size, (list, tuple))
        kernel = torch.Tensor(list(x.shape[2:])) // torch.Tensor(
            [list(layer.output_size)]
        )
        kernel_ops = torch.prod(kernel)
        flops = kernel_ops * y.numel()

    # linear layer:
    elif layer_type in ["Linear"]:
        weight_ops = layer.weight.numel()
        flops = x.size()[0] * weight_ops

    # batch normalization / layer normalization:
    elif layer_type in [
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "SyncBatchNorm",
        "LayerNorm",
    ]:
        # batchnorm can be merged into conv op. Thus, count 0 FLOPS
        flops = 0

    # dropout layer
    elif layer_type in ["Dropout"]:
        flops = 0

    elif layer_type == "Identity":
        flops = 0

    elif hasattr(layer, "flops"):
        # If the module already defines a method to compute flops with the signature
        # below, we use it to compute flops
        #
        #   Class MyModule(nn.Module):
        #     def flops(self, x):
        #       ...
        flops = layer.flops(x)

    if flops is None:
        raise ClassyProfilerNotImplementedError(layer)

    message = [
        f"module type: {typestr}",
        f"input size: {get_shape(x)}",
        f"output size: {get_shape(y)}",
        f"params(M): {count_params(layer) / 1e6}",
        f"flops(M): {int(flops) / 1e6}",
    ]
    if verbose:
        logging.info("\t".join(message))
    return flops


def _layer_activations(
    layer: nn.Module, x: Any, out: Any, verbose: bool = False
) -> int:
    """
    Computes the number of activations produced by a single layer.

    Activations are counted only for convolutional layers. To override this behavior, a
    layer can define a method to compute activations with the signature below, which
    will be used to compute the activations instead.

    Class MyModule(nn.Module):
        def activations(self, x, out):
            ...
    """
    typestr = layer.__repr__()
    if hasattr(layer, "activations"):
        activations = layer.activations(x, out)
    elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        activations = out.numel()
    else:
        return 0

    message = [f"module: {typestr}", f"activations: {activations}"]
    if verbose:
        logging.info("\t".join(message))
    return activations


def summarize_profiler_info(prof: torch.autograd.profiler.profile) -> str:
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


class ComplexityComputer:
    def __init__(self, compute_fn: Callable, count_unique: bool, verbose: bool = False):
        self.compute_fn = compute_fn
        self.count_unique = count_unique
        self.count = 0
        self.verbose = verbose
        self.seen_modules = set()

    def compute(self, layer: nn.Module, x: Any, out: Any, module_name: str):
        if self.count_unique and module_name in self.seen_modules:
            return
        if self.verbose:
            logging.info(f"module name: {module_name}")
        self.count += self.compute_fn(layer, x, out, self.verbose)
        self.seen_modules.add(module_name)

    def reset(self):
        self.count = 0
        self.seen_modules.clear()


def _patched_computation_module(
    module: nn.Module, complexity_computer: ComplexityComputer, module_name: str
):
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


def modify_forward(
    model: nn.Module,
    complexity_computer: ComplexityComputer,
    prefix: str = "",
    patch_attr: str = None,
) -> nn.Module:
    """
    Modify forward pass to measure a module's parameters, like FLOPs.
    """
    # Recursively update all the modules in the model. A module is patched if it
    # contains the patch_attr (like the flops() function for FLOPs computation) or it is
    # a leaf. We stop recursing if we patch a module since that module is supposed
    # to return the results for all its children as well.
    # Since this recursion can lead to the same module being patched through different
    # paths, we make sure we only patch un-patched modules.
    if hasattr(model, "orig_type"):
        return model
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


def restore_forward(model: nn.Module, patch_attr: str = None) -> nn.Module:
    """
    Restore original forward in model.
    """
    for module in model.modules():
        if hasattr(module, "orig_type"):
            # module has been patched; un-patch it
            module.__class__ = module.orig_type
    return model


def compute_complexity(
    model: nn.Module,
    compute_fn: Callable,
    input_shape: Tuple[int],
    input_key: Optional[Union[str, List[str]]] = None,
    patch_attr: str = None,
    compute_unique: bool = False,
    verbose: bool = False,
) -> int:
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

    complexity_computer = ComplexityComputer(compute_fn, compute_unique, verbose)

    # measure FLOPs:
    modify_forward(model, complexity_computer, patch_attr=patch_attr)
    try:
        # compute complexity in eval mode
        with eval_model(model), torch.no_grad():
            model.forward(input)
    finally:
        restore_forward(model, patch_attr=patch_attr)

    return complexity_computer.count


def compute_flops(
    model: nn.Module,
    input_shape: Tuple[int] = (3, 224, 224),
    input_key: Optional[Union[str, List[str]]] = None,
) -> int:
    """
    Compute the number of FLOPs needed for a forward pass.
    """
    return compute_complexity(
        model, _layer_flops, input_shape, input_key, patch_attr="flops", verbose=True
    )


def compute_activations(
    model: nn.Module,
    input_shape: Tuple[int] = (3, 224, 224),
    input_key: Optional[Union[str, List[str]]] = None,
) -> int:
    """
    Compute the number of activations created in a forward pass.
    """
    return compute_complexity(
        model,
        _layer_activations,
        input_shape,
        input_key,
        patch_attr="activations",
        verbose=True,
    )


def count_params(model: nn.Module) -> int:
    """
    Count the number of parameters in a model.
    """
    assert isinstance(model, nn.Module)
    return sum((parameter.nelement() for parameter in model.parameters()))
