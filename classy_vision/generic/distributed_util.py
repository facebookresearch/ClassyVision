#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


# Default to GPU 0
_cuda_device_index: int = 0

# Setting _cuda_device_index to -1 internally implies that we should use CPU
_CPU_DEVICE_INDEX = -1


def convert_to_distributed_tensor(tensor):
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)


def convert_to_normal_tensor(tensor, orig_device):
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()
    return tensor


def is_distributed_training_run():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )


def is_master():
    """
    Returns True if this is rank 0 of a distributed training job OR if it is
    a single trainer job. Otherwise False.
    """
    return get_rank() == 0


def all_reduce_mean(tensor):
    """
    Wrapper over torch.distributed.all_reduce for performing mean reduction
    of tensor over all processes.
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.all_reduce(tensor, torch.distributed.ReduceOp.SUM)
        tensor = tensor / torch.distributed.get_world_size()
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


def all_reduce_sum(tensor):
    """
    Wrapper over torch.distributed.all_reduce for performing sum
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.all_reduce(tensor, torch.distributed.ReduceOp.SUM)
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


def gather_tensors_from_all(tensor):
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]

    return gathered_tensors


def gather_from_all(tensor):
    gathered_tensors = gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def barrier():
    """
    Wrapper over torch.distributed.barrier, returns without waiting
    if the distributed process group is not initialized instead of throwing error.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    torch.distributed.barrier()


def get_world_size():
    """
    Simple wrapper for correctly getting worldsize in both distributed
    / non-distributed settings
    """
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )


def get_rank():
    """
    Simple wrapper for correctly getting rank in both distributed
    / non-distributed settings
    """
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )


def set_cuda_device_index(idx: int):
    global _cuda_device_index
    _cuda_device_index = idx
    torch.cuda.set_device(_cuda_device_index)


def set_cpu_device():
    global _cuda_device_index
    _cuda_device_index = _CPU_DEVICE_INDEX


def get_cuda_device_index() -> int:
    return _cuda_device_index


def init_distributed_data_parallel_model(model, broadcast_buffers=False):
    global _cuda_device_index

    if _cuda_device_index == _CPU_DEVICE_INDEX:
        # CPU-only model, don't specify device
        return torch.nn.parallel.DistributedDataParallel(
            model, broadcast_buffers=broadcast_buffers
        )
    else:
        # GPU model
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[_cuda_device_index],
            output_device=_cuda_device_index,
            broadcast_buffers=broadcast_buffers,
        )
