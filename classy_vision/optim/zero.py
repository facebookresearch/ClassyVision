#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch.distributed as dist
from classy_vision.generic.distributed_util import get_primary_rank
from classy_vision.generic.util import recursive_copy_to_device
from classy_vision.optim.classy_optimizer import OptionsView
from torch.optim.optimizer import Optimizer

from . import ClassyOptimizer, build_optimizer, register_optimizer


try:
    from fairscale.optim.oss import OSS

    fairscale_available = True
except ImportError:
    fairscale_available = False


if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


@register_optimizer("zero")
class ZeRO(ClassyOptimizer):
    def __init__(self, base_optimizer_config: Dict[str, Any]):
        """Wraps an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>`
        optimizer and shards its state as described by ZeRO_.
        ::
            opt = OSS(params, optim=torch.optim.Adam, lr=0.01)

        .. _ZeRO: https://arxiv.org/abs/1910.02054

        This instance holds all of the parameters for the model (in the .param_groups attribute)
        but relies on a wrapped optimizer, which only process an original shard of the parameters.
        Every step all the parameters are synced across the replicas.
        """

        assert (
            fairscale_available
        ), "The Fairscale library needs to be installed to use this optimizer. See https://github.com/facebookresearch/fairscale"

        self.options_view = None
        self.optimizer = None
        self._param_group_schedulers = None
        self.base_optimizer_config = base_optimizer_config

    def prepare(self, param_groups) -> None:
        # ClassyOptimizer have deferred initialization, while OSS needs access to the
        # raw optimizer instance, hence the trampoline

        def optimizer_constructor(param_groups: _params_t, *args, **kwargs):
            logging.info("Building a ZeRO enabled optimizer")
            base_classy_optimizer = build_optimizer(self.base_optimizer_config)
            base_classy_optimizer.prepare(param_groups)
            return base_classy_optimizer.optimizer

        self.optimizer = OSS(
            params=param_groups, optim=optimizer_constructor, group=dist.group.WORLD
        )

        self.options_view = OptionsView(self.optimizer)

        # Copy the optimizer-shard specific keys which would be missing in the wrap
        # but may be used by external schedulers
        for pg in self.optimizer.param_groups:
            for k, v in self.optimizer.optim.param_groups[0].items():
                if k not in pg.keys():
                    pg[k] = v

    @classmethod
    def from_config(cls, config):
        return cls(base_optimizer_config=config["base_optimizer"])

    def on_epoch(self, where: float) -> None:
        # Run the normal LR schedulers
        super().on_epoch(where)

        # Materialize the optimizer state on the replica in charge of checkpointing
        if where > 0.0:
            logging.info(
                "Consolidating sharded state on primary rank. Where: %d" % where
            )
            self.consolidate_state_dict()

    def consolidate_state_dict(self) -> None:
        self.optimizer.consolidate_state_dict(recipient_rank=get_primary_rank())
