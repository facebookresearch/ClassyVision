#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from enum import Enum

import torch.nn as nn

from .classy_block import ClassyBlock


class ClassyModelEvaluationMode(Enum):
    DEFAULT = 0
    VIDEO_CLIP_AVERAGING = 1


class ClassyModel(nn.Module):
    """Base class for models in classy vision.

    A model refers either to a specific architecture (e.g. ResNet50) or a
    family of architectures (e.g. ResNet). Models can take arguments in the
    constructor in order to configure different behavior (e.g.
    hyperparameters).  Classy Models must implement :method:`from_config` in
    order to allow instantiation from a configuration file. Like regular
    PyTorch models, Classy Models must also implement :method:`forward`, where
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

    @classmethod
    def from_config(cls, config):
        """Implemented by children.

        This is a factory method for generating the class from a config.

        Args:
            config (Dict): Contains params for constructing the model
        """
        raise NotImplementedError

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
                head.unique_id: head.state_dict() for head in heads.values()
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
        for block, head_states in state["model"]["heads"].items():
            self._attachable_blocks[block].load_head_states(head_states)

    def set_classy_state(self, state):
        """Set the state of the ClassyModel.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :method:`get_classy_state`.

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
        applying the final fc layer.
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
        for block in self._attachable_blocks.values():
            block.set_heads([])

    def set_heads(self, heads):
        """Attach all the heads to corresponding blocks.

        A head is expected to be a ClassyHead object. For more
        details, see :class:`ClassyHead`.

        Args:
            heads (Dict): a mapping between attachable block name
                and a dictionary of heads attached to that block. For
                example, if you have two different teams that want to
                attach two different heads for downstream classifiers to
                the 15th block, then they would use:
                heads = {"block15":
                    {"team1": classifier_head1, "team2": classifier_head2}
                }

        """
        self._clear_heads()

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
        """Returns the heads on the model

        Function returns the heads a dictionary of block names to
        nn.modules attached to that block.

        """
        all_heads = {}
        for name, block in self._attachable_blocks.items():
            heads = block.get_heads()
            if len(heads) > 0:
                all_heads[name] = heads
        return all_heads

    @property
    def head_outputs(self):
        """Return outputs of all heads in the format of Dict[head_id, output]

        Head outputs are cached during a forward pass.
        """
        outputs = {}
        for blk in self._attachable_blocks.values():
            outputs.update(blk.head_outputs)
        return outputs

    def get_optimizer_params(self, bn_weight_decay=False):
        """Returns param groups for optimizer.

        Function to return dict of params with "keys" from
        {"regularized_params", "unregularized_params"}
        to "values" a list of torch Params.

        "weight_decay" provided as part of optimizer is only used
        for "regularized_params". For "unregularized_params", weight_decay is set
        to 0.0

        This implementation sets BatchNorm's all trainable params to be
        unregularized_params if bn_weight_decay is False.

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

        TODO: Remove this once we have a video task, this logic should
        live in a video specific task

        """
        return ClassyModelEvaluationMode.DEFAULT
