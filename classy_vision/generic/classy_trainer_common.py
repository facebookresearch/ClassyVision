#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


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
