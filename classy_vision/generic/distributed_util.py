#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io

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
    return all_reduce_op(
        tensor,
        torch.distributed.ReduceOp.SUM,
        lambda t: t / torch.distributed.get_world_size(),
    )


def all_reduce_sum(tensor):
    """
    Wrapper over torch.distributed.all_reduce for performing sum
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.SUM)


def all_reduce_min(tensor):
    """
    Wrapper over torch.distributed.all_reduce for performing min
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.MIN)


def all_reduce_op(tensor, op, after_op_func=None):
    """
    Wrapper over torch.distributed.all_reduce for performing
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.all_reduce(tensor, op)
        if after_op_func is not None:
            tensor = after_op_func(tensor)
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


def broadcast(tensor, src=0):
    """
    Wrapper over torch.distributed.broadcast for broadcasting a tensor from the source
    to all processes in both distributed / non-distributed scenarios.
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.broadcast(tensor, src)
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


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


def init_distributed_data_parallel_model(
    model, broadcast_buffers=False, find_unused_parameters=True
):
    global _cuda_device_index

    if _cuda_device_index == _CPU_DEVICE_INDEX:
        # CPU-only model, don't specify device
        return torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        # GPU model
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[_cuda_device_index],
            output_device=_cuda_device_index,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
        )


def broadcast_object(obj):
    if is_master():
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)])
        length_tensor = broadcast(length_tensor)
        data_tensor = torch.ByteTensor(data)
        data_tensor = broadcast(data_tensor)
    else:
        length_tensor = torch.LongTensor([0])
        length_tensor = broadcast(length_tensor)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8)
        data_tensor = broadcast(data_tensor)
        buffer = io.BytesIO(data_tensor.numpy())
        obj = torch.load(buffer)
    return obj
