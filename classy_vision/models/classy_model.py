#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from enum import Enum
from typing import Any, Dict

import torch
import torch.nn as nn
from classy_vision.heads.classy_head import ClassyHead

from .classy_block import ClassyBlock


class ClassyModelEvaluationMode(Enum):
    DEFAULT = 0
    VIDEO_CLIP_AVERAGING = 1


class ClassyModel(nn.Module):
    """Base class for models in classy vision.

    A model refers either to a specific architecture (e.g. ResNet50) or a
    family of architectures (e.g. ResNet). Models can take arguments in the
    constructor in order to configure different behavior (e.g.
    hyperparameters).  Classy Models must implement :func:`from_config` in
    order to allow instantiation from a configuration file. Like regular
    PyTorch models, Classy Models must also implement :func:`forward`, where
    the bulk of the inference logic lives.

    Classy Models also have some advanced functionality for production
    fine-tuning systems. For example, we allow users to train a trunk
    model and then attach heads to the model via the attachable
    blocks.  Making your model support the trunk-heads paradigm is
    completely optional.

    """

    def __init__(self):
        """Constructor for ClassyModel."""
        super().__init__()

        self._attachable_blocks = {}
        self._heads = nn.ModuleDict()
        self._head_outputs = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassyModel":
        """Instantiates a ClassyModel from a configuration.

        Args:
            config: A configuration for the ClassyModel.

        Returns:
            A ClassyModel instance.
        """
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, checkpoint):
        from . import build_model

        model = build_model(checkpoint["input_args"]["config"]["model"])
        model.set_classy_state(checkpoint["classy_state_dict"]["base_model"])
        return model

    def get_classy_state(self, deep_copy=False):
        """Get the state of the ClassyModel.

        The returned state is used for checkpointing.

        Args:
            deep_copy: If True, creates a deep copy of the state Dict. Otherwise, the
                returned Dict's state will be tied to the object's.

        Returns:
            A state dictionary containing the state of the model.
        """
        # If the model doesn't have head for fine-tuning, all of model's state
        # live in the trunk
        attached_heads = self.get_heads()
        # clear heads to get trunk only states. There shouldn't be any component
        # states depend on heads
        self._clear_heads()
        trunk_state_dict = super().state_dict()
        self.set_heads(attached_heads)

        head_state_dict = {}
        for block, heads in attached_heads.items():
            head_state_dict[block] = {
                head_name: head.state_dict() for head_name, head in heads.items()
            }
        model_state_dict = {
            "model": {"trunk": trunk_state_dict, "heads": head_state_dict}
        }
        if deep_copy:
            model_state_dict = copy.deepcopy(model_state_dict)
        return model_state_dict

    def load_head_states(self, state):
        """Load only the state (weights) of the heads.

        For a trunk-heads model, this function allows the user to
        only update the head state of the model. Useful for attaching
        fine-tuned heads to a pre-trained trunk.

        Args:
            state (Dict): Contains the classy model state under key "model"

        """
        for block_name, head_states in state["model"]["heads"].items():
            for head_name, head_state in head_states.items():
                self._heads[block_name][head_name].load_state_dict(head_state)

    def set_classy_state(self, state):
        """Set the state of the ClassyModel.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the model from a checkpoint.
        """
        self.load_head_states(state)

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
        applying the final fully connected layer.
        """
        return self.forward(x)

    def build_attachable_block(self, name, module):
        """
        Add a wrapper to the module to allow to attach heads to the module.
        """
        if name in self._attachable_blocks:
            raise ValueError("Found duplicated block name {}".format(name))
        block = ClassyBlock(name, module)
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
        self._heads.clear()
        self._head_outputs.clear()

    def set_heads(self, heads: Dict[str, Dict[str, ClassyHead]]):
        """Attach all the heads to corresponding blocks.

        A head is expected to be a ClassyHead object. For more
        details, see :class:`classy_vision.heads.ClassyHead`.

        Args:
            heads (Dict): a mapping between attachable block name
                and a dictionary of heads attached to that block. For
                example, if you have two different teams that want to
                attach two different heads for downstream classifiers to
                the 15th block, then they would use:

                .. code-block:: python

                  heads = {"block15":
                      {"team1": classifier_head1, "team2": classifier_head2}
                  }
        """
        self._clear_heads()

        head_ids = set()
        for block_name, block_heads in heads.items():
            if block_name not in self._attachable_blocks:
                raise ValueError(
                    "block {} does not exist or can not be attached".format(block_name)
                )
            self._attachable_blocks[block_name].set_cache_output()
            for head in block_heads.values():
                if head.unique_id in head_ids:
                    raise ValueError("head id {} already exists".format(head.unique_id))
                head_ids.add(head.unique_id)
            self._heads[block_name] = nn.ModuleDict(block_heads)

    def get_heads(self):
        """Returns the heads on the model

        Function returns the heads a dictionary of block names to
        `nn.Modules <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_
        attached to that block.

        """
        return {block_name: dict(heads) for block_name, heads in self._heads.items()}

    @property
    def head_outputs(self):
        """Return outputs of all heads in the format of Dict[head_id, output]

        Head outputs are cached during a forward pass.
        """
        return self._head_outputs.copy()

    def get_block_outputs(self) -> Dict[str, torch.Tensor]:
        outputs = {}
        for name, block in self._attachable_blocks.items():
            outputs[name] = block.output
        return outputs

    def execute_heads(self) -> Dict[str, torch.Tensor]:
        block_outs = self.get_block_outputs()
        outputs = {}
        for block_name, heads in self._heads.items():
            for head in heads.values():
                outputs[head.unique_id] = head(block_outs[block_name])
        self._head_outputs = outputs
        return outputs

    def get_optimizer_params(self, bn_weight_decay=False):
        """Returns param groups for optimizer.

        Function to return dict of params with "keys" from
        {"regularized_params", "unregularized_params"}
        to "values" a list of `pytorch Params <https://pytorch.org/docs/
        stable/nn.html#torch.nn.Parameter>`_.

        "weight_decay" provided as part of optimizer is only used
        for "regularized_params". For "unregularized_params", weight_decay is set
        to 0.0

        This implementation sets `BatchNorm's <https://pytorch.org/docs/
        stable/nn.html#normalization-layers>`_ all trainable params to be
        unregularized_params if ``bn_weight_decay`` is False.

        Override this function for any custom behavior.

        Args:
            bn_weight_decay (bool): Apply weight decay to bn params if true
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
            elif not bn_weight_decay and isinstance(
                module,
                (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
            ):
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
        """If implemented, returns expected input tensor shape
        """
        raise NotImplementedError

    @property
    def output_shape(self):
        """If implemented, returns expected output tensor shape
        """
        raise NotImplementedError

    @property
    def model_depth(self):
        """If implemented, returns number of layers in model
        """
        raise NotImplementedError

    @property
    def evaluation_mode(self):
        """Used by video models for averaging over contiguous clips.
        """
        # TODO: Remove this once we have a video task, this logic should
        # live in a video specific task
        return ClassyModelEvaluationMode.DEFAULT
