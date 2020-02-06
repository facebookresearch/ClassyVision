#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import contextlib
import json
import logging
import math
import os
import sys
import traceback
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from fvcore.common.file_io import PathManager
from torch._six import container_abcs


# constants:
CHECKPOINT_FILE = "checkpoint.torch"
CPU_DEVICE = torch.device("cpu")


def is_pos_int(number):
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


def is_pos_float(number):
    """
    Returns True if a number is a positive float.
    """
    return type(number) == float and number >= 0.0


def is_pos_int_list(l):
    """
    Returns True if a list contains positive integers
    """
    return type(l) == list and all(is_pos_int(n) for n in l)


def is_long_tensor(tensor):
    """
    Returns True if a tensor is a long tensor.
    """
    if torch.is_tensor(tensor):
        return tensor.type().endswith("LongTensor")
    else:
        return False


def is_float_tensor(tensor):
    """
    Returns True if a tensor is a float tensor.
    """
    if torch.is_tensor(tensor):
        return tensor.type().endswith("FloatTensor")
    else:
        return False


def is_double_tensor(tensor):
    """
    Returns True if a tensor is a double tensor.
    """
    if torch.is_tensor(tensor):
        return tensor.type().endswith("DoubleTensor")
    else:
        return False


def is_leaf(module):
    """
    Returns True if module is leaf in the graph.
    """
    assert isinstance(module, nn.Module), "module should be nn.Module"
    return len([c for c in module.children()]) == 0 or hasattr(module, "_mask")


def is_on_gpu(model):
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


def is_not_none(sample):
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


def copy_upvalue(value, upvalue):
    """
    Iteratively copies a particular value into an upvalue dict.
    """
    assert type(value) == type(upvalue), "type of value and upvalue must match"
    if type(value) == dict:
        upvalue.clear()
        for key, val in value.items():
            upvalue[key] = val
    elif type(value) == list:
        del upvalue[:]
        for _, val in value:
            upvalue.append(val)
    else:
        raise BaseException("unsupported upvalue type")


def recursive_copy_to_gpu(value, non_blocking=True, max_depth=3, curr_depth=0):
    """
    Recursively searches lists, tuples, dicts and copies to GPU if possible.
    Note:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the GPU.
    """
    if curr_depth >= max_depth:
        raise ValueError("Depth of value object is too deep")

    try:
        return value.cuda(non_blocking=non_blocking)
    except AttributeError:
        if isinstance(value, container_abcs.Sequence):
            gpu_val = []
            for val in value:
                gpu_val.append(
                    recursive_copy_to_gpu(
                        val,
                        non_blocking=non_blocking,
                        max_depth=max_depth,
                        curr_depth=curr_depth + 1,
                    )
                )

            return gpu_val if isinstance(value, list) else tuple(gpu_val)
        elif isinstance(value, container_abcs.Mapping):
            gpu_val = {}
            for key, val in value.items():
                gpu_val[key] = recursive_copy_to_gpu(
                    val,
                    non_blocking=non_blocking,
                    max_depth=max_depth,
                    curr_depth=curr_depth + 1,
                )

            return gpu_val

        raise AttributeError("Value must have .cuda attr or be a Seq / Map iterable")


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
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


def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy in a multi-class problem from an NxK output
    matrix and a corresponding Nx1 target matrix with values in 0, ..., K-1.
    """

    # assertions:
    assert torch.is_tensor(output)
    assert torch.is_tensor(target)
    assert output.size(0) == target.size(0)
    if type(topk) == int:
        topk = (topk,)
    assert type(topk) == tuple or type(topk) == list
    assert all(is_pos_int(k) for k in topk)
    maxk = max(topk)
    assert maxk < output.size(1)

    # determine whether predictions are correct:
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # compute accuracies:
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.size(0)).item())
    return res


def binary_accuracy(output, target, threshold=0.0):
    """
    Computes the accuracy based on a real-valued output tensor and a corresponding
    target matrix of the same size with valuesin [0, 1].

    An optional threshold for positive classification can be specified (default = 0).
    """

    # assertions:
    assert torch.is_tensor(output)
    assert torch.is_tensor(target)
    assert output.size() == target.size()

    # compute accuracy:
    correct = output.ge(threshold).type_as(target).eq_(target).long()
    return correct.sum().item() * 100.0 / float(target.nelement())


def create_class_histograms(pred_prob, target, num_bins):
    """
    Creates two histograms to contain total positive or true samples
    for a given class at a given score value and total samples at a
    given predicted probability.  Then for a given threshold we can compute
    precision = true_samples_above_threshold /
    total_samples_above_threshold and recall =
    true_samples_above_threshold / true_samples. Output is expected to
    be prediction probabilities.

    The pred_prob vector is num_classes x batchsize_per_replica, the target vector is
    num_classes x 1, should contain integers corresponding to each class.
    Output is two tensors of same size: num_bins x num_classes
    where the bins are equispaced from 0 to 1 across the score.
    The class_hist[:, c] contains true example histogram for class c
    prediction probabilities while total_hist[:, c] contains histogram
    for all samples for class c prediction probabilities.
    """

    # assertions:
    assert torch.is_tensor(pred_prob)
    assert torch.is_tensor(target)
    assert pred_prob.size()[0] == target.size()[0], "%s pred_prob, %s target" % (
        str(pred_prob.size()),
        str(target.size()),
    )
    assert (
        pred_prob.lt(0.0).sum() == 0 and pred_prob.gt(1.0).sum() == 0
    ), "Prediction probability must be between 0 and 1"

    num_classes = pred_prob.size()[1]
    class_hist = torch.zeros([num_bins, num_classes], dtype=torch.int64)
    total_hist = torch.zeros([num_bins, num_classes], dtype=torch.int64)
    for c in range(num_classes):
        total_hist[:, c] = torch.histc(
            pred_prob[:, c], bins=num_bins, min=0.0, max=1.0
        ).long()
        class_hist[:, c] = torch.histc(
            pred_prob[:, c][(target == c).nonzero().squeeze(1)],
            bins=num_bins,
            min=0.0,
            max=1.0,
        ).long()

    return class_hist, total_hist


def _find_last_larger_than(target, val_array):
    """
    Takes an array and finds the last value larger than the target value.
    Returns the index of that value, returns -1 if none exists in array.
    """
    ind = -1
    for j in range(len(val_array), 0, -1):
        if val_array[j - 1] > target:
            ind = j - 1
            break
    return ind


def calc_ap(prec, recall):
    """
    Computes average precision from precision recall curves. Curves
    are expected to be same size and 1D.
    """
    assert prec.size() == recall.size(), "Precision and recall curves must be same size"
    ap = 0.0
    if len(prec) == 0:
        return ap
    prev_r = 0.0
    prev_p = prec[0]
    for p, r in zip(prec, recall):
        ap += (r - prev_r) * (p + prev_p) / 2.0
        prev_r = r
        prev_p = p

    return ap


def compute_pr_curves(class_hist, total_hist):
    """
    Computes precision recall curves from the true sample / total
    sample histogram tensors. The histogram tensors are num_bins x num_classes
    and each column represents a histogram over
    prediction_probabilities.

    The two tensors should have the same dimensions.
    The two tensors should have nonnegative integer values.

    Returns map of precision / recall values from highest precision to lowest
    and the calculated AUPRC (i.e. the average precision).
    """
    assert torch.is_tensor(class_hist) and torch.is_tensor(
        total_hist
    ), "Both arguments must be tensors"
    assert (
        class_hist.dtype == torch.int64 and total_hist.dtype == torch.int64
    ), "Both arguments must contain int64 values"
    assert (
        len(class_hist.size()) == 2 and len(total_hist.size()) == 2
    ), "Both arguments must have 2 dimensions, (score_bin, class)"
    assert (
        class_hist.size() == total_hist.size()
    ), """
        For compute_pr_curve, arguments must be  of same size.
        class_hist.size(): %s
        total_hist.size(): %s
        """ % (
        str(class_hist.size()),
        str(total_hist.size()),
    )
    assert (
        class_hist > total_hist
    ).sum() == 0, (
        "Invalid. Class histogram must be less than or equal to total histogram"
    )

    num_bins = class_hist.size()[0]
    # Cumsum from highest bucket to lowest
    cum_class_hist = torch.cumsum(torch.flip(class_hist, dims=(0,)), dim=0).double()
    cum_total_hist = torch.cumsum(torch.flip(total_hist, dims=(0,)), dim=0).double()
    class_totals = cum_class_hist[-1, :]

    prec_t = cum_class_hist / cum_total_hist
    recall_t = cum_class_hist / class_totals

    prec = torch.unbind(prec_t, dim=1)
    recall = torch.unbind(recall_t, dim=1)
    assert len(prec) == len(
        recall
    ), "The number of precision curves does not match the number of recall curves"

    final_prec = []
    final_recall = []
    final_ap = []
    for c, prec_curve in enumerate(prec):
        recall_curve = recall[c]
        assert (
            recall_curve.size()[0] == num_bins and prec_curve.size()[0] == num_bins
        ), "Precision and recall curves do not have the correct number of entries"

        # Check if any samples from class were seen
        if class_totals[c] == 0:
            continue

        # Remove duplicate entries
        prev_r = torch.tensor(-1.0).double()
        prev_p = torch.tensor(1.1).double()
        new_recall_curve = torch.tensor([], dtype=torch.double)
        new_prec_curve = torch.tensor([], dtype=torch.double)
        for idx, r in enumerate(recall_curve):
            p = prec_curve[idx]
            # Remove points on PR curve that are invalid
            if r.item() <= 0:
                continue

            # Remove duplicates (due to empty buckets):
            if r.item() == prev_r.item() and p.item() == prev_p.item():
                continue

            # Add points to curve
            new_recall_curve = torch.cat((new_recall_curve, r.unsqueeze(0)), dim=0)
            new_prec_curve = torch.cat((new_prec_curve, p.unsqueeze(0)), dim=0)
            prev_r = r
            prev_p = p

        ap = calc_ap(new_prec_curve, new_recall_curve)
        final_prec.append(new_prec_curve)
        final_recall.append(new_recall_curve)
        final_ap.append(ap)

    return {"prec": final_prec, "recall": final_recall, "ap": final_ap}


def get_checkpoint_dict(task, input_args, deep_copy=False):
    assert isinstance(
        input_args, dict
    ), f"Unexpected input_args of type: {type(input_args)}"
    return {
        "input_args": input_args,
        "classy_state_dict": task.get_classy_state(deep_copy=deep_copy),
    }


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


def update_classy_model(model, model_state_dict, reset_heads):
    """
    Updates the model with the provided model state dictionary.

    Args:
        model: ClassyVisionModel instance to update
        model_state_dict: State dict, should be the output of a call to
            ClassyVisionModel.get_classy_state().
        reset_heads: if False, uses the heads' state from model_state_dict.
    """
    try:
        if reset_heads:
            current_model_state_dict = model.get_classy_state()
            # replace the checkpointed head states with source head states
            model_state_dict["model"]["heads"] = current_model_state_dict["model"][
                "heads"
            ]
        model.set_classy_state(model_state_dict)
        logging.info("Model state load successful")
        return True
    except Exception:
        logging.exception("Could not load the model state")
    return False


def update_classy_state(task, state_dict):
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
    Saves a state variable to the specified checkpoint folder. Returns filename
    of checkpoint if successful, and False otherwise.
    """

    # make sure that we have a checkpoint folder:
    if not PathManager.isdir(checkpoint_folder):
        try:
            PathManager.mkdirs(checkpoint_folder)
        except BaseException:
            logging.warning(
                "Could not create folder %s." % checkpoint_folder, exc_info=True
            )
    if not PathManager.isdir(checkpoint_folder):
        return False

    # write checkpoint atomically:
    try:
        full_filename = f"{checkpoint_folder}/{checkpoint_file}"
        with PathManager.open(full_filename, "wb") as f:
            torch.save(state, f)
        return full_filename
    except BaseException:
        logging.warning(
            "Did not write checkpoint to %s." % checkpoint_folder, exc_info=True
        )
        return False


def stepwise_learning_rate(base_lr, epoch, optimizer, epoch_step=30):
    """
    Step-wise reduction of learning rate by a factor of 10 every epoch_step epochs.
    """
    lr = base_lr * (0.1 ** (epoch // epoch_step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def cosine_learning_rate(base_lr, epoch, optimizer, max_epoch=90):
    """
    Cosine-shaped reduction of learning rate.
    """
    lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * epoch / max_epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def visualize_image(img, mean=None, std=None):
    """
    Make preprocessed image look pretty again to facilitate visualization.
    """

    # assertions:
    assert torch.is_tensor(img) and img.dim() == 3
    for val in [mean, std]:
        if val is not None:
            assert torch.is_tensor(val)
            assert val.dim() == 1 and len(val) == img.size(0)

    # undo mean/std normalization:
    new_img = img.clone()
    if mean is not None and std is not None:
        for c in range(new_img.size(0)):
            new_img[c].mul_(std[c]).add_(mean[c])

    # normalize image to be a byte image:
    if new_img.max() < 1.1:
        new_img.mul_(255.0)
    return new_img.byte()


def set_proxies():
    """Set proxies to allow downloading of external URLs."""
    os.environ["HTTP_PROXY"] = "http://fwdproxy:8080"
    os.environ["HTTPS_PROXY"] = "http://fwdproxy:8080"
    os.environ["http_proxy"] = "fwdproxy:8080"
    os.environ["https_proxy"] = "fwdproxy:8080"


def unset_proxies():
    """Unset proxies to prevent downloading of external URLs."""
    del os.environ["HTTP_PROXY"]
    del os.environ["HTTPS_PROXY"]
    del os.environ["http_proxy"]
    del os.environ["https_proxy"]


def flatten_dict(value_dict, prefix="", sep="_"):
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
def torch_seed(seed):
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


# NOTE [ Python Traceback Reference Cycle Problem ]
#
# When using sys.exc_info(), it is important to **not** store the exc_info[2],
# which is the traceback, because otherwise you will run into the traceback
# reference cycle problem, i.e., the traceback holding reference to the frame,
# and the frame (which holds reference to all the object in its temporary scope)
# holding reference the traceback.
#
# TODO: aadcock: This is a fork of the PyTorch ExceptionWrapper class
# to facilitate the backfill dataloader until we kill it. Once we kill
# datasets/core/backfill_async_dataset_iterator.py we can kill these functions
class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    def __repr__(self):
        return self


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = "Caught {} {}.\nOriginal {}".format(
            self.exc_type.__name__, self.where, self.exc_msg
        )
        if self.exc_type == KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        raise self.exc_type(msg)


def convert_to_one_hot(targets, classes):
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


def maybe_convert_to_one_hot(target, model_output):
    """
    This function infers whether target is integer or 0/1 encoded
    and converts it to 0/1 encoding if necessary.
    """
    target_shape_list = list(target.size())

    if len(target_shape_list) == 1 or (
        len(target_shape_list) == 2 and target_shape_list[1] == 1
    ):
        target = convert_to_one_hot(target.view(-1, 1), model_output.shape[1])

    assert (target.shape == model_output.shape) and (
        torch.min(target.eq(0) + target.eq(1)) == 1
    ), (
        "Target must be one-hot/multi-label encoded and of the "
        "same shape as model_output."
    )

    return target


def bind_method_to_class(method, cls):
    """
    Binds an already bound method to the provided class.
    """
    return method.__func__.__get__(cls)


def get_model_dummy_input(
    model, input_shape, input_key, batchsize=1, non_blocking=False
):
    if isinstance(input_key, list):
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
