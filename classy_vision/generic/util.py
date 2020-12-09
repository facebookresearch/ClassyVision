#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import contextlib
import json
import logging
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from classy_vision.generic.distributed_util import broadcast_object, is_primary
from fvcore.common.file_io import PathManager
from torch._six import container_abcs


try:
    import apex

    apex_available = True
except ImportError:
    apex_available = False


# constants:
CHECKPOINT_FILE = "checkpoint.torch"
CPU_DEVICE = torch.device("cpu")
GPU_DEVICE = torch.device("cuda")


def is_pos_int(number: int) -> bool:
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


def is_pos_float(number: float) -> bool:
    """
    Returns True if a number is a positive float.
    """
    return type(number) == float and number >= 0.0


def is_pos_int_list(l: List) -> bool:
    """
    Returns True if a list contains positive integers
    """
    return type(l) == list and all(is_pos_int(n) for n in l)


def is_pos_int_tuple(t: Tuple) -> bool:
    """
    Returns True if a tuple contains positive integers
    """
    return type(t) == tuple and all(is_pos_int(n) for n in t)


def is_long_tensor(tensor: torch.Tensor) -> bool:
    """
    Returns True if a tensor is a long tensor.
    """
    if torch.is_tensor(tensor):
        return tensor.type().endswith("LongTensor")
    else:
        return False


def is_float_tensor(tensor: torch.Tensor) -> bool:
    """
    Returns True if a tensor is a float tensor.
    """
    if torch.is_tensor(tensor):
        return tensor.type().endswith("FloatTensor")
    else:
        return False


def is_double_tensor(tensor: torch.Tensor) -> bool:
    """
    Returns True if a tensor is a double tensor.
    """
    if torch.is_tensor(tensor):
        return tensor.type().endswith("DoubleTensor")
    else:
        return False


def is_leaf(module: nn.Module) -> bool:
    """
    Returns True if module is leaf in the graph.
    """
    assert isinstance(module, nn.Module), "module should be nn.Module"
    return len(list(module.children())) == 0 or hasattr(module, "_mask")


def is_on_gpu(model: torch.nn.Module) -> bool:
    """
    Returns True if all parameters of a model live on the GPU.
    """
    assert isinstance(model, torch.nn.Module)
    on_gpu = True
    has_params = False
    for param in model.parameters():
        has_params = True
        if not param.data.is_cuda:
            on_gpu = False
    return has_params and on_gpu


def is_not_none(sample: Any) -> bool:
    """
    Returns True if sample is not None and constituents are not none.
    """
    if sample is None:
        return False

    if isinstance(sample, (list, tuple)):
        if any(s is None for s in sample):
            return False

    if isinstance(sample, dict):
        if any(s is None for s in sample.values()):
            return False
    return True


def copy_model_to_gpu(model, loss=None):
    """
    Copies a model and (optional) loss to GPU and enables cudnn benchmarking.
    For multiple gpus training, the model in DistributedDataParallel for
    distributed training.
    """
    if not torch.backends.cudnn.deterministic:
        torch.backends.cudnn.benchmark = True
    model = model.cuda()

    if loss is not None:
        loss = loss.cuda()
        return model, loss
    else:
        return model


def recursive_copy_to_device(
    value: Any, non_blocking: bool, device: torch.device
) -> Any:
    """
    Recursively searches lists, tuples, dicts and copies any object which
    supports an object.to API (e.g. tensors) to device if possible.
    Other values are passed as-is in the result.

    Note:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """
    if isinstance(value, list) or isinstance(value, tuple):
        device_val = []
        for val in value:
            device_val.append(
                recursive_copy_to_device(val, non_blocking=non_blocking, device=device)
            )

        return device_val if isinstance(value, list) else tuple(device_val)
    elif isinstance(value, container_abcs.Mapping):
        device_val = {}
        for key, val in value.items():
            device_val[key] = recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device
            )

        return device_val
    elif callable(getattr(value, "to", None)):
        return value.to(device=device, non_blocking=non_blocking)

    return value


def recursive_copy_to_gpu(value: Any, non_blocking: bool = True) -> Any:
    """
    Recursively searches lists, tuples, dicts and copies tensors to GPU if
    possible. Non-tensor values are passed as-is in the result.
    Note:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the GPU.
    """
    return recursive_copy_to_device(
        value=value, non_blocking=non_blocking, device=GPU_DEVICE
    )


@contextlib.contextmanager
def numpy_seed(seed: Optional[int], *addl_seeds: int) -> None:
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_checkpoint_dict(
    task, input_args: Optional[Dict], deep_copy: bool = False
) -> Dict[str, Any]:
    assert input_args is None or isinstance(
        input_args, dict
    ), f"Unexpected input_args of type: {type(input_args)}"
    return {
        "input_args": input_args,
        "classy_state_dict": task.get_classy_state(deep_copy=deep_copy),
    }


def load_and_broadcast_checkpoint(
    checkpoint_path: str, device: torch.device = CPU_DEVICE
) -> Optional[Dict]:
    """Loads a checkpoint on master and broadcasts it to all replicas.

    This is a collective operation which needs to be run in sync on all replicas.

    See :func:`load_checkpoint` for the arguments.
    """
    if is_primary():
        checkpoint = load_checkpoint(checkpoint_path, device)
    else:
        checkpoint = None
    logging.info(f"Broadcasting checkpoint loaded from {checkpoint_path}")
    return broadcast_object(checkpoint)


def load_checkpoint(
    checkpoint_path: str, device: torch.device = CPU_DEVICE
) -> Optional[Dict]:
    """Loads a checkpoint from the specified checkpoint path.

    Args:
        checkpoint_path: The path to load the checkpoint from. Can be a file or a
            directory. If it is a directory, the checkpoint is loaded from
            :py:data:`CHECKPOINT_FILE` inside the directory.
        device: device to load the checkpoint to

    Returns:
        The checkpoint, if it exists, or None.
    """
    if not checkpoint_path:
        return None

    assert device is not None, "Please specify what device to load checkpoint on"
    assert device.type in ["cpu", "cuda"], f"Unknown device: {device}"
    if device.type == "cuda":
        assert torch.cuda.is_available()

    if not PathManager.exists(checkpoint_path):
        logging.warning(f"Checkpoint path {checkpoint_path} not found")
        return None
    if PathManager.isdir(checkpoint_path):
        checkpoint_path = f"{checkpoint_path.rstrip('/')}/{CHECKPOINT_FILE}"

    if not PathManager.exists(checkpoint_path):
        logging.warning(f"Checkpoint file {checkpoint_path} not found.")
        return None

    logging.info(f"Attempting to load checkpoint from {checkpoint_path}")
    # load model on specified device and not on saved device for model and return
    # the checkpoint
    with PathManager.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=device)
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def update_classy_model(
    model, model_state_dict: Dict, reset_heads: bool, strict: bool = True
) -> bool:
    """
    Updates the model with the provided model state dictionary.

    Args:
        model: ClassyVisionModel instance to update
        model_state_dict: State dict, should be the output of a call to
            ClassyVisionModel.get_classy_state().
        reset_heads: if False, uses the heads' state from model_state_dict.
        strict: if True, strictly match the module/buffer keys in current model and
            pass-in model_state_dict
    """
    try:
        if reset_heads:
            current_model_state_dict = model.get_classy_state()
            # replace the checkpointed head states with source head states
            model_state_dict["model"]["heads"] = current_model_state_dict["model"][
                "heads"
            ]
        model.set_classy_state(model_state_dict, strict=strict)
        logging.info("Model state load successful")
        return True
    except Exception:
        logging.exception("Could not load the model state")
    return False


def update_classy_state(task, state_dict: Dict) -> bool:
    """
    Updates the task with the provided task dictionary.

    Args:
        task: ClassyTask instance to update
        state_dict: State dict, should be the output of a call to
            ClassyTask.get_classy_state().
    """
    logging.info("Loading classy state from checkpoint")

    try:
        task.set_classy_state(state_dict)
        logging.info("Checkpoint load successful")
        return True
    except Exception:
        logging.exception("Could not load the checkpoint")

    return False


def save_checkpoint(checkpoint_folder, state, checkpoint_file=CHECKPOINT_FILE):
    """
    Saves a state variable to the specified checkpoint folder. Returns the filename
    of the checkpoint if successful. Raises an exception otherwise.
    """

    # make sure that we have a checkpoint folder:
    if not PathManager.isdir(checkpoint_folder):
        try:
            PathManager.mkdirs(checkpoint_folder)
        except BaseException:
            logging.warning("Could not create folder %s." % checkpoint_folder)
            raise

    # write checkpoint atomically:
    try:
        full_filename = f"{checkpoint_folder}/{checkpoint_file}"
        with PathManager.open(full_filename, "wb") as f:
            torch.save(state, f)
        return full_filename
    except BaseException:
        logging.warning(
            "Unable to write checkpoint to %s." % checkpoint_folder, exc_info=True
        )
        raise


def flatten_dict(value_dict: Dict, prefix="", sep="_") -> Dict:
    """
    Flattens nested dict into (key, val) dict. Used for flattening meters
    structure, so that they can be visualized.
    """
    items = []
    for k, v in value_dict.items():
        key = prefix + sep + k if prefix else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(value_dict=v, prefix=key, sep=sep).items())
        else:
            items.append((key, v))
    return dict(items)


def load_json(json_path):
    """
    Loads a json config from a file.
    """
    assert os.path.exists(json_path), "Json file %s not found" % json_path
    json_file = open(json_path)
    json_config = json_file.read()
    json_file.close()
    try:
        config = json.loads(json_config)
    except BaseException as err:
        raise Exception("Failed to validate config with error: %s" % str(err))

    return config


@contextlib.contextmanager
def torch_seed(seed: Optional[int]):
    """Context manager which seeds the PyTorch PRNG with the specified seed and
    restores the state afterward. Setting seed to None is equivalent to running
    the code without the context manager."""
    if seed is None:
        yield
        return
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)


def convert_to_one_hot(targets: torch.Tensor, classes) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    """
    assert (
        torch.max(targets).item() < classes
    ), "Class Index must be less than number of classes"
    one_hot_targets = torch.zeros(
        (targets.shape[0], classes), dtype=torch.long, device=targets.device
    )
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


def maybe_convert_to_one_hot(
    target: torch.Tensor, model_output: torch.Tensor
) -> torch.Tensor:
    """
    This function infers whether target is integer or 0/1 encoded
    and converts it to 0/1 encoding if necessary.
    """
    target_shape_list = list(target.size())

    if len(target_shape_list) == 1 or (
        len(target_shape_list) == 2 and target_shape_list[1] == 1
    ):
        target = convert_to_one_hot(target.view(-1, 1), model_output.shape[1])

    # target are not necessarily hard 0/1 encoding. It can be soft
    # (i.e. fractional) in some cases, such as mixup label
    assert (
        target.shape == model_output.shape
    ), "Target must of the same shape as model_output."

    return target


def get_model_dummy_input(
    model,
    input_shape: Any,
    input_key: Union[str, List[str]],
    batchsize: int = 1,
    non_blocking: bool = False,
) -> Any:

    # input_shape with type dict of dict
    # e.g. {"key_1": {"key_1_1": [2, 3], "key_1_2": [4, 5, 6], "key_1_3": []}
    if isinstance(input_shape, dict):
        input = {}
        for key, value in input_shape.items():
            input[key] = get_model_dummy_input(
                model, value, input_key, batchsize, non_blocking
            )
    elif isinstance(input_key, list):
        # av mode, with multiple input keys
        input = {}
        for i, key in enumerate(input_key):
            shape = (batchsize,) + tuple(input_shape[i])
            cur_input = torch.zeros(shape)
            if next(model.parameters()).is_cuda:
                cur_input = cur_input.cuda(non_blocking=non_blocking)
            input[key] = cur_input
    else:
        # add a dimension to represent minibatch axis
        shape = (batchsize,) + tuple(input_shape)
        input = torch.zeros(shape)
        if next(model.parameters()).is_cuda:
            input = input.cuda(non_blocking=non_blocking)
        if input_key:
            input = {input_key: input}
    return input


def get_batchsize_per_replica(x: Union[Tuple, List, Dict]) -> int:
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


def split_batchnorm_params(model: nn.Module):
    """Finds the set of BatchNorm parameters in the model.

    Recursively traverses all parameters in the given model and returns a tuple
    of lists: the first element is the set of batchnorm parameters, the second
    list contains all other parameters of the model."""
    batchnorm_params = []
    other_params = []
    for module in model.modules():
        # If module has children (i.e. internal node of constructed DAG) then
        # only add direct parameters() to the list of params, else go over
        # children node to find if they are BatchNorm or have "bias".
        if list(module.children()) != []:
            for params in module.parameters(recurse=False):
                if params.requires_grad:
                    other_params.append(params)
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            for params in module.parameters():
                if params.requires_grad:
                    batchnorm_params.append(params)
        else:
            for params in module.parameters():
                if params.requires_grad:
                    other_params.append(params)
    return batchnorm_params, other_params


class Timer:
    """Timer context manager to get the elapsed time for a code block.

    Example:
        .. code-block:: python

            with Timer() as timer:
                do_something()
            elapsed_time = timer.elapsed_time
    """

    def __init__(self):
        self.start = 0
        self.elapsed_time = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        self.elapsed_time = time.perf_counter() - self.start


@contextlib.contextmanager
def _train_mode(model: nn.Module, train_mode: bool):
    """Context manager which sets the train mode of a model. After returning, it
    restores the state of every sub-module individually."""
    train_modes = {}
    for name, module in model.named_modules():
        train_modes[name] = module.training
    try:
        model.train(train_mode)
        yield
    finally:
        for name, module in model.named_modules():
            module.training = train_modes[name]


def log_class_usage(component_type, klass):
    """This function is used to log the usage of different Classy components."""
    identifier = "ClassyVision"
    if klass and hasattr(klass, "__name__"):
        identifier += f".{component_type}.{klass.__name__}"
    torch._C._log_api_usage_once(identifier)


def get_torch_version():
    return torch.__version__[:3]


train_model = partial(_train_mode, train_mode=True)
train_model.__doc__ = """Context manager which puts the model in train mode.

    After returning, it restores the state of every sub-module individually.
    """


eval_model = partial(_train_mode, train_mode=False)
eval_model.__doc__ = """Context manager which puts the model in eval mode.

    After returning, it restores the state of every sub-module individually.
    """


def master_params(optimizer):
    """Generator to iterate over all parameters in the optimizer param_groups.

    When apex is available, uses that to guarantee we get the FP32 copy of the
    parameters when O2 is enabled. Otherwise, iterate ourselves."""
    if apex_available:
        yield from apex.amp.master_params(optimizer)
    else:
        for group in optimizer.param_groups:
            for p in group["params"]:
                yield p
