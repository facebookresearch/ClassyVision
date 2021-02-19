#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any

import torch.distributed as dist
from classy_vision.generic.distributed_util import get_primary_rank
from classy_vision.optim import ClassyOptimizer, build_optimizer, register_optimizer


try:
    from fairscale.optim.oss import OSS

    fairscale_available = True
except ImportError:
    fairscale_available = False


@register_optimizer("zero")
class ZeRO(ClassyOptimizer):
    def __init__(self, base_optimizer: ClassyOptimizer):
        """Wraps an arbitrary :class:`ClassyOptimizer <classy_vision.optim.ClassyOptimizer>`
        optimizer and shards its state as described by ZeRO_.
        ::
            opt = OSS(params, optim=torch.optim.Adam, lr=0.01)

        .. _ZeRO: https://arxiv.org/abs/1910.02054

        This instance holds all of the parameters for the model (in the .param_groups attribute)
        but relies on a wrapped optimizer, which only process an original shard of the parameters.
        Every step all the parameters are synced across the replicas. The Fairscale library is used
        https://github.com/facebookresearch/fairscale
        """

        assert (
            fairscale_available
        ), "The Fairscale library needs to be installed to use this optimizer."

        super().__init__()
        self.base_optimizer = base_optimizer

    def prepare(self, param_groups) -> None:
        assert (
            dist.is_initialized()
        ), "torch.distributed needs to be initialized to prepare this rank"

        def optimizer_constructor(param_groups: Any, *args, **kwargs):
            # ClassyOptimizer have deferred initialization, while OSS needs access to the
            # raw optimizer instance, hence the trampoline
            logging.debug("Building a ZeRO enabled optimizer")
            self.base_optimizer.prepare(param_groups)
            return self.base_optimizer.optimizer

        self.optimizer = OSS(params=param_groups, optim=optimizer_constructor)

    @classmethod
    def from_config(cls, config):
        return cls(base_optimizer=build_optimizer(config["base_optimizer"]))

    def on_epoch(self, where: float) -> None:
        # Run the normal LR schedulers
        super().on_epoch(where)

        # Materialize the optimizer state on the replica in charge of checkpointing
        logging.info("Consolidating sharded state on primary rank. Where: %d" % where)
        self.consolidate_state_dict()

    def consolidate_state_dict(self) -> None:
        self.optimizer.consolidate_state_dict(recipient_rank=get_primary_rank())
