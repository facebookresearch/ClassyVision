#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Callable, Dict, List, Union

import torch
from classy_vision.hooks import ClassyHook, ClassyHookFunctions
from classy_vision.state.classy_state import ClassyState

from .distributed_util import all_reduce_mean
from .perf_stats import PerfTimer
from .util import recursive_copy_to_gpu


def run_hooks(
    state: ClassyState,
    local_variables: Dict[str, Any],
    hooks: List[ClassyHook],
    hook_function: str,
) -> None:
    """
    Helper function that runs the hook_function for all the classy hooks.
    """
    for hook in hooks:
        getattr(hook, hook_function)(state, local_variables)


def _remove_dummy_samples_from_batch(temp_vals):
    """
    If 'is_dummy_sample' key exists then return only real sample's
    model_output and target.
    """
    model_output = temp_vals["output"]
    target = temp_vals["sample"]["target"]
    if "is_dummy_sample" in temp_vals["sample"]:
        model_output = model_output.index_select(
            dim=0,
            index=(temp_vals["sample"]["is_dummy_sample"] != 1.0).nonzero().squeeze(1),
        )
        target = target.index_select(
            dim=0,
            index=(temp_vals["sample"]["is_dummy_sample"] != 1.0).nonzero().squeeze(1),
        )
        return model_output, target
    return model_output, target


def train_step(state, hooks, use_gpu, local_variables=None):
    assert isinstance(state, ClassyState)

    if local_variables is None:
        local_variables = {}

    # We'll time train_step and some of its sections, and accumulate values
    # into perf_stats if it were defined in local_variables:
    perf_stats = local_variables.get("perf_stats", None)
    timer_train_step = PerfTimer("train_step_total", perf_stats)
    timer_train_step.start()

    # Process next sample
    with PerfTimer("read_sample", perf_stats):
        sample = next(state.get_data_iterator())
        local_variables["sample"] = sample

        assert (
            isinstance(local_variables["sample"], dict)
            and "input" in local_variables["sample"]
            and "target" in local_variables["sample"]
        ), "Returned sample [{}] is not a map with 'input' and 'target' keys".format(
            local_variables["sample"]
        )

    run_hooks(state, local_variables, hooks, ClassyHookFunctions.on_sample.name)

    # Copy sample to GPU
    local_variables["target"] = local_variables["sample"]["target"]
    if use_gpu:
        for key, value in local_variables["sample"].items():
            local_variables["sample"][key] = recursive_copy_to_gpu(
                value, non_blocking=True
            )

    # Only need gradients during training
    context = torch.enable_grad() if state.train else torch.no_grad()
    with context:
        # Forward pass
        with PerfTimer("forward", perf_stats):
            local_variables["output"] = state.model(local_variables["sample"]["input"])

        # Only use non-dummy samples for finding loss and meters.
        model_output, target = _remove_dummy_samples_from_batch(local_variables)

        run_hooks(state, local_variables, hooks, ClassyHookFunctions.on_forward.name)

        # If all the samples in the batch are dummy then use loss of 0.0
        # We still need to backprop though as all the processes sync on
        # that.
        if model_output.shape[0] == 0:
            local_variables["local_loss"] = torch.autograd.Variable(
                torch.tensor(0.0, device=target.device), requires_grad=True
            )
        else:
            local_variables["local_loss"] = state.criterion(model_output, target)

        # NOTE: This performs an all_reduce_mean() on the losses across the replicas.
        # The reduce should ideally be weighted by the length of the targets on each
        # replica. This will only be an issue when there are dummy samples present
        # (once an epoch) and will only impact the loss reporting (slightly).
        with PerfTimer("loss_allreduce", perf_stats):
            local_variables["loss"] = local_variables["local_loss"].detach().clone()
            local_variables["loss"] = all_reduce_mean(local_variables["loss"])

        state.losses.append(
            local_variables["loss"].data.cpu().item()
            * local_variables["target"].size(0)
        )

        run_hooks(state, local_variables, hooks, ClassyHookFunctions.on_loss.name)

        model_output_cpu = model_output.cpu() if use_gpu else model_output

        # Update meters
        with PerfTimer("meters_update", perf_stats):
            for meter in state.meters:
                meter.update(model_output_cpu, target.detach().cpu())

    num_samples_in_step = state.get_global_batchsize()
    state.num_samples_this_phase += num_samples_in_step

    # For training phases, run backwards pass / update optimizer
    if state.train:
        with PerfTimer("backward", perf_stats):
            state.optimizer.backward(local_variables["local_loss"])

        run_hooks(state, local_variables, hooks, ClassyHookFunctions.on_backward.name)

        state.optimizer.update_schedule_on_step(state.where)
        with PerfTimer("optimizer_step", perf_stats):
            state.optimizer.step()

        run_hooks(state, local_variables, hooks, ClassyHookFunctions.on_update.name)

        state.num_updates += num_samples_in_step

    timer_train_step.stop()
    timer_train_step.record()

    return state
