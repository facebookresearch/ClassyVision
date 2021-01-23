#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from classy_vision.generic.util import log_class_usage
from classy_vision.heads.classy_head import ClassyHead

from .classy_block import ClassyBlock


class _ClassyModelMeta(type):
    """Metaclass to return a ClassyModel instance wrapped by a ClassyModelWrapper."""

    def __call__(cls, *args, **kwargs):
        """Override the __call__ function for the metaclass.

        This is called when a new instance of a class with this class as its metaclass
        is initialized. For example -

        .. code-block:: python
          class MyClass(metaclass=_ClassyModelMeta):
              wrapper_cls = MyWrapper

          my_class_instance = MyClass()  # returned instance will be a MyWrapper
        """
        classy_model = super().__call__(*args, **kwargs)

        wrapper_cls = cls.wrapper_cls
        if wrapper_cls is not None:
            # wrap the ClassyModel instance with a wrapper class and return that instead
            classy_model = wrapper_cls(classy_model)
        return classy_model


class _ClassyModelMethod:
    """Class to override ClassyModel method calls to ensure the wrapper is returned.

    This helps override calls like model.cuda() which return self, to return the
    wrapper instead of the underlying classy_model.
    """

    def __init__(self, wrapper, classy_method):
        self.wrapper = wrapper
        self.classy_method = classy_method

    def __call__(self, *args, **kwargs):
        ret_val = self.classy_method(*args, **kwargs)
        if ret_val is self.wrapper.classy_model:
            # if the method is returning the classy_model, return the wrapper instead
            ret_val = self.wrapper
        return ret_val


class ClassyModelWrapper:
    """Base ClassyModel wrapper class.

    This class acts as a thin pass through wrapper which lets users modify the behavior
    of ClassyModels, such as changing the return output of the forward() call.
    This wrapper acts as a ClassyModel by itself and the underlying model can be
    accessed by the `classy_model` attribute.
    """

    # TODO: Make this torchscriptable by inheriting from nn.Module / ClassyModel

    def __init__(self, classy_model):
        self.classy_model = classy_model

    def __getattr__(self, name):
        if name != "classy_model" and hasattr(self, "classy_model"):
            attr = getattr(self.classy_model, name)
            if isinstance(attr, types.MethodType):
                attr = _ClassyModelMethod(self, attr)
            return attr
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        # __setattr__ works differently from __getattr__ and is called even when the
        # attribute is a method, like forward.
        if name not in ["classy_model", "forward"] and hasattr(self, "classy_model"):
            setattr(self.classy_model, name, value)
        else:
            super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        return self.classy_model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f"Classy {type(self.classy_model)}:\n{self.classy_model.__repr__()}"

    @property
    def __class__(self):
        return self.classy_model.__class__


class ClassyModelHeadExecutorWrapper(ClassyModelWrapper):
    """Wrapper which changes the forward to also execute and return head output."""

    def forward(self, *args, **kwargs):
        out = self.classy_model(*args, **kwargs)

        if len(self._heads) == 0:
            return out

        # heads have been attached to the model, return their output instead
        head_outputs = self.execute_heads()
        if len(head_outputs) == 1:
            return list(head_outputs.values())[0]
        else:
            return head_outputs


class ClassyModel(nn.Module, metaclass=_ClassyModelMeta):
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

    NOTE: Advanced users can modify the behavior of their implemented models by
        specifying the `wrapper_cls` class attribute, which should be a class
        derived from :class:`ClassyModelWrapper` (see the documentation for that class
        for more information). Users can set it to `None` to skip wrapping their model
        and to make their model torchscriptable. This is set to
        :class:`ClassyModelHeadExecutorWrapper` by default.
    """

    wrapper_cls = ClassyModelHeadExecutorWrapper

    _attachable_block_names: List[str]
    __jit_unused_properties__ = ["attachable_block_names", "head_outputs"]

    def __init__(self):
        """Constructor for ClassyModel."""
        super().__init__()
        self._attachable_blocks = {}
        self._attachable_block_names = []
        self._heads = nn.ModuleDict()
        self._head_outputs = {}

        log_class_usage("Model", self.__class__)

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
    def from_model(
        cls,
        model: nn.Module,
        input_shape: Optional[Tuple] = None,
        model_depth: Optional[int] = None,
    ):
        """Converts an :class:`nn.Module` to a `ClassyModel`.

        Args:
            model: The model to convert
            For the remaining args, look at the corresponding properties of ClassyModel

        Returns:
            A ClassyModel instance.
        """
        return _ClassyModelAdapter(
            model, input_shape=input_shape, model_depth=model_depth
        )

    @classmethod
    def from_checkpoint(cls, checkpoint):
        from . import build_model

        model = build_model(checkpoint["input_args"]["config"]["model"])
        model.set_classy_state(checkpoint["classy_state_dict"]["base_model"])
        return model

    def get_classy_state(self, deep_copy=False):
        """Get the state of the ClassyModel.

        The returned state is used for checkpointing.

        NOTE: For advanced users, the structure of the returned dict is -
            `{"model": {"trunk": trunk_state, "heads": heads_state}}`.
            The trunk state is the state of the model when no heads are attached.

        Args:
            deep_copy: If True, creates a deep copy of the state Dict. Otherwise, the
                returned Dict's state will be tied to the object's.

        Returns:
            A state dictionary containing the state of the model.
        """
        attached_heads = self.get_heads()
        # clear heads to get the state of the model without any heads, which we refer to
        # as the trunk state. If the model doesn't have heads attached, all of the
        # model's state lives in the trunk.
        self.clear_heads()
        trunk_state_dict = self.state_dict()
        self.set_heads(attached_heads)

        head_state_dict = {}
        for block, heads in attached_heads.items():
            head_state_dict[block] = {
                head.unique_id: head.state_dict() for head in heads
            }
        model_state_dict = {
            "model": {"trunk": trunk_state_dict, "heads": head_state_dict}
        }
        if deep_copy:
            model_state_dict = copy.deepcopy(model_state_dict)
        return model_state_dict

    def load_head_states(self, state, strict=True):
        """Load only the state (weights) of the heads.

        For a trunk-heads model, this function allows the user to
        only update the head state of the model. Useful for attaching
        fine-tuned heads to a pre-trained trunk.

        Args:
            state (Dict): Contains the classy model state under key "model"

        """
        for block_name, head_states in state["model"]["heads"].items():
            for head_name, head_state in head_states.items():
                self._heads[block_name][head_name].load_state_dict(head_state, strict)

    def set_classy_state(self, state, strict=True):
        """Set the state of the ClassyModel.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the model from a checkpoint.
        """
        # load the state for heads
        self.load_head_states(state, strict)

        # clear the heads to set the trunk's state. This is done because when heads are
        # attached to modules, we wrap them by ClassyBlocks, thereby changing the
        # structure of the model and its state dict. So, the trunk state is always
        # fetched / set when there are no blocks attached.
        attached_heads = self.get_heads()
        self.clear_heads()
        self.load_state_dict(state["model"]["trunk"], strict)

        # set the heads back again
        self.set_heads(attached_heads)

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

    def _build_attachable_block(self, name, module):
        """
        Add a wrapper to the module to allow to attach heads to the module.
        """
        if name in self._attachable_blocks:
            raise ValueError("Found duplicated block name {}".format(name))
        block = ClassyBlock(name, module)
        self._attachable_blocks[name] = block
        self._attachable_block_names.append(name)
        return block

    @property
    def attachable_block_names(self):
        """
        Return names of all attachable blocks.
        """
        return self._attachable_block_names

    def clear_heads(self):
        # clear all existing heads
        self._heads.clear()
        self._head_outputs.clear()
        self._strip_classy_blocks(self)
        self._attachable_blocks = {}
        self._attachable_block_names = []

    def _strip_classy_blocks(self, module):
        for name, child_module in module.named_children():
            if isinstance(child_module, ClassyBlock):
                module.add_module(name, child_module.wrapped_module())
            self._strip_classy_blocks(child_module)

    def _make_module_attachable(self, module, module_name):
        found = False
        for name, child_module in module.named_children():
            if name == module_name:
                module.add_module(
                    name, self._build_attachable_block(name, child_module)
                )
                found = True
                # do not exit - we will check all possible modules and raise an
                # exception if there are duplicates
            found_in_child = self._make_module_attachable(child_module, module_name)
            found = found or found_in_child
        return found

    def set_heads(self, heads: Dict[str, List[ClassyHead]]):
        """Attach all the heads to corresponding blocks.

        A head is expected to be a ClassyHead object. For more
        details, see :class:`classy_vision.heads.ClassyHead`.

        Args:
            heads (Dict): a mapping between attachable block name
                and a list of heads attached to that block. For
                example, if you have two different teams that want to
                attach two different heads for downstream classifiers to
                the 15th block, then they would use:

                .. code-block:: python

                  heads = {"block15":
                      [classifier_head1, classifier_head2]
                  }
        """
        self.clear_heads()

        head_ids = set()
        for block_name, block_heads in heads.items():
            if not self._make_module_attachable(self, block_name):
                raise KeyError(f"{block_name} not found in the model")
            for head in block_heads:
                if head.unique_id in head_ids:
                    raise ValueError("head id {} already exists".format(head.unique_id))
                head_ids.add(head.unique_id)
            self._heads[block_name] = nn.ModuleDict(
                {head.unique_id: head for head in block_heads}
            )

    def get_heads(self):
        """Returns the heads on the model

        Function returns the heads a dictionary of block names to
        `nn.Modules <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_
        attached to that block.

        """
        return {
            block_name: list(heads.values())
            for block_name, heads in self._heads.items()
        }

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

    @property
    def input_shape(self):
        """Returns the input shape that the model can accept, excluding the batch dimension.

        By default it returns (3, 224, 224).
        """
        return (3, 224, 224)


class _ClassyModelAdapter(ClassyModel):
    """
    Class which adapts an `nn.Module <https://pytorch.org/docs/stable/
    nn.html#torch.nn.Module>`_ to a ClassyModel by wrapping the model.

    The only required argument is the model, the additional args are needed
    to get some additional capabilities from Classy Vision to work.
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple] = None,
        model_depth: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self._input_shape = input_shape
        self._model_depth = model_depth

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        if hasattr(self.model, "extract_features"):
            return self.model.extract_features(x)
        return super().extract_features(x)

    @property
    def input_shape(self):
        if self._input_shape is not None:
            return self._input_shape
        return super().input_shape
