#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.optim
from classy_vision.generic.util import is_pos_float
from torch.optim import Optimizer

from . import ClassyOptimizer, register_optimizer


class RMSpropTFOptimizer(Optimizer):
    r"""Implements RMSprop algorithm.

    NOTE: This code is copied from :class:`torch.optim.RMSProp`, with the epsilon
    moved inside the square root to match tensorflow's implementation.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The
    effective learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where
    :math:`\alpha` is the scheduled learning rate and :math:`v` is the weighted moving
    average of the squared gradient.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the square root in the denominator to
            improve numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = (
                        square_avg.addcmul(-1, grad_avg, grad_avg)
                        .add_(group["eps"])
                        .sqrt_()
                    )
                else:
                    avg = square_avg.add_(group["eps"]).sqrt_()

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    p.data.add_(-group["lr"], buf)
                else:
                    p.data.addcdiv_(-group["lr"], grad, avg)

        return loss


@register_optimizer("rmsprop_tf")
class RMSPropTF(ClassyOptimizer):
    def __init__(
        self,
        lr: float = 0.1,
        momentum: float = 0,
        weight_decay: float = 0,
        alpha: float = 0.99,
        eps: float = 1e-8,
        centered: bool = False,
    ) -> None:
        super().__init__()

        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._alpha = alpha
        self._eps = eps
        self._centered = centered

    def prepare(self, param_groups):
        self.optimizer = RMSpropTFOptimizer(
            param_groups,
            lr=self._lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
            alpha=self._alpha,
            eps=self._eps,
            centered=self._centered,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RMSPropTF":
        """Instantiates a RMSPropTF from a configuration.

        Args:
            config: A configuration for a RMSPropTF.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A RMSPropTF instance.
        """
        # Default params
        config.setdefault("lr", 0.1)
        config.setdefault("momentum", 0.0)
        config.setdefault("weight_decay", 0.0)
        config.setdefault("alpha", 0.99)
        config.setdefault("eps", 1e-8)
        config.setdefault("centered", False)

        for key in ["momentum", "alpha"]:
            assert (
                config[key] >= 0.0 and config[key] < 1.0 and type(config[key]) == float
            ), f"Config must contain a '{key}' in [0, 1) for RMSPropTF optimizer"
        assert is_pos_float(
            config["eps"]
        ), f"Config must contain a positive 'eps' for RMSPropTF optimizer"
        assert isinstance(
            config["centered"], bool
        ), "Config must contain a boolean 'centered' param for RMSPropTF optimizer"

        return cls(
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            alpha=config["alpha"],
            eps=config["eps"],
            centered=config["centered"],
        )
