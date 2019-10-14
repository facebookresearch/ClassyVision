#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import torch.nn as nn

from .classy_module import ClassyModule


class ClassyVisionModel(nn.Module):
    def __init__(self, num_classes, freeze_trunk=False):
        super().__init__()

        self._num_classes = num_classes
        self._attachable_blocks = {}
        self.freeze_trunk = freeze_trunk

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    @property
    def num_classes(self):
        # Flatten the dictionary of dictionaries into a list of heads
        heads = [head for heads in self.get_heads().values() for head in heads.values()]

        if len(heads) == 1:
            return heads[0].num_classes
        elif len(heads) >= 1:
            logging.error("Tried to get num_classes on a model with multiple heads")
            raise RuntimeError
        return self._num_classes

    def get_classy_state(self, deep_copy=False):
        """
        Returns a dictionary containing the state stored inside the object.

        If deep_copy is True (default False), creates a deep copy. Otherwise,
        the returned dict's attributes will be tied to the object's.
        """
        # If the model doesn't have head for fine-tuning, all of model's state
        # live in the trunk
        attached_heads = self.get_heads()
        # clear heads to get trunk only states. There shouldn't be any component
        # states depend on heads
        self._clear_heads()
        trunk_state_dict = super().state_dict()
        self.set_heads(attached_heads, self.freeze_trunk)

        head_state_dict = {}
        for block, heads in attached_heads.items():
            head_state_dict[block] = {
                head.unique_id: head.state_dict() for head in heads.values()
            }
        model_state_dict = {
            "model": {"trunk": trunk_state_dict, "heads": head_state_dict}
        }
        if deep_copy:
            model_state_dict = copy.deepcopy(model_state_dict)
        return model_state_dict

    def set_classy_state(self, state):
        for block, head_states in state["model"]["heads"].items():
            self._attachable_blocks[block].load_head_states(head_states)

        current_state = self.state_dict()
        current_state.update(state["model"]["trunk"])
        super().load_state_dict(current_state)

    def forward(self, x):
        """
        Perform computation of blocks in the order define in get_blocks.
        """
        raise NotImplementedError

    def extract_features(self, x):
        """
        Extract features from the model.

        Derived classes can implement this method to extract the features before
        applying the final fc layer.
        """
        return self.forward(x)

    # TODO (changhan): was planning to re-implement pytorch-summary but it's
    # based on a dummy forward pass. Will leave it to a separate diff
    def summarize(self):
        raise NotImplementedError

    def build_attachable_block(self, name, module):
        """
        Add a wrapper to the module to allow to attach heads to the module.
        """
        if name in self._attachable_blocks:
            raise ValueError("Found duplicated block name {}".format(name))
        block = ClassyModule(name, module)
        self._attachable_blocks[name] = block
        return block

    @property
    def attachable_block_names(self):
        """
        Return names of all attachable blocks.
        """
        return self._attachable_blocks.keys()

    def _clear_heads(self):
        # clear all existing heads
        for block in self._attachable_blocks.values():
            block.set_heads([])

    def set_heads(self, heads, freeze_trunk):
        """
        Attach all the heads to corresponding blocks.
        A head is a neural network which takes input from an interior block of
        another model and produces an output to be used externally from the
        model / model trunk.
        heads -- a mapping between fork block name and dictionary of heads
                 (e.g. {"block15": {"team1": head1, "team2": head2}})
        freeze_trunk -- whether freeze the parameters of layers in the base model
        """
        self._clear_heads()

        if freeze_trunk:
            for param in self.parameters():
                param.requires_grad = False

        head_ids = set()
        for block_name, heads in heads.items():
            for head in heads.values():
                if head.unique_id in head_ids:
                    raise ValueError("head id {} already exists".format(head.unique_id))
                head_ids.add(head.unique_id)
            if block_name not in self._attachable_blocks:
                raise ValueError(
                    "block {} does not exist or can not be attached".format(block_name)
                )
            self._attachable_blocks[block_name].set_heads(heads.values())

    def get_heads(self):
        all_heads = {}
        for name, block in self._attachable_blocks.items():
            heads = block.get_heads()
            if len(heads) > 0:
                all_heads[name] = heads
        return all_heads

    @property
    def head_outputs(self):
        """
        Return outputs of all heads in the format of dict<head_id, output>
        """
        outputs = {}
        for blk in self._attachable_blocks.values():
            outputs.update(blk.head_outputs)
        return outputs

    def validate(self, dataset_output_shape):
        raise NotImplementedError

    def get_optimizer_params(self):
        """
        Function to return dict of params with "keys" from
        {"regularized_params", "unregularized_params"}
        to "values" a list of torch Params.

        "weight_decay" provided as part of optimizer is only used
        for "regularized_params". For "unregularized_params", weight_decay is set
        to 0.0

        This implementation sets BatchNorm's all trainable params to be
        unregularized_params.

        Override this function for any custom behavior.
        """
        unregularized_params = []
        regularized_params = []
        for module in self.modules():
            # If module has children (i.e. internal node of constructed DAG) then
            # only add direct parameters() to the list of params, else go over
            # children node to find if they are BatchNorm or have "bias".
            if list(module.children()) != []:
                for params in module.parameters(recurse=False):
                    if params.requires_grad:
                        regularized_params.append(params)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for params in module.parameters():
                    if params.requires_grad:
                        unregularized_params.append(params)
            else:
                for params in module.parameters():
                    if params.requires_grad:
                        regularized_params.append(params)
        return {
            "regularized_params": regularized_params,
            "unregularized_params": unregularized_params,
        }

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def model_depth(self):
        raise NotImplementedError
