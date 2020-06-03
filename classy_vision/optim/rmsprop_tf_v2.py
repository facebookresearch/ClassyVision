#!/usr/bin/env python3

# A different implementation of TensorFlow RMSprop optimization.
# Code forked from mobile-vision/projects/classification_pytorch/lib/optimizers/rmsprop_tf.py

import logging
from typing import Any, Dict

import torch
import torch.optim
from classy_vision.generic.util import is_pos_float
from torch.optim import Optimizer

from . import ClassyOptimizer, register_optimizer


class RMSpropTFV2Optimizer(Optimizer):
    """Implements RMSprop algorithm (TensorFlow style epsilon)
    NOTE: This is a direct cut-and-paste of PyTorch RMSprop with eps applied before sqrt
    to closer match Tensorflow for matching hyper-params.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing (decay) constant (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_decay (bool, optional): decoupled weight decay as per
            (https://arxiv.org/abs/1711.05101)
        lr_in_momentum (bool, optional): learning rate scaling is included in the
            momentum buffer update as per defaults in Tensorflow
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.9,
        eps=1e-10,
        weight_decay=0,
        momentum=0.0,
        centered=False,
        decoupled_decay=False,
        lr_in_momentum=True,
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
            decoupled_decay=decoupled_decay,
            lr_in_momentum=lr_in_momentum,
        )

        super(RMSpropTFV2Optimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropTFV2Optimizer, self).__setstate__(state)
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
                    # PyTorch inits to zero
                    state["square_avg"] = torch.ones_like(p.data)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p.data)

                square_avg = state["square_avg"]
                one_minus_alpha = 1.0 - group["alpha"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    if "decoupled_decay" in group and group["decoupled_decay"]:
                        p.data.add_(-group["weight_decay"], p.data)
                    else:
                        grad = grad.add(group["weight_decay"], p.data)

                # Tensorflow order of ops for updating squared avg
                square_avg.add_(one_minus_alpha, grad.pow(2) - square_avg)
                # PyTorch original
                # square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.add_(one_minus_alpha, grad - grad_avg)
                    # PyTorch original
                    # grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    # eps moved in sqrt
                    avg = (
                        square_avg.addcmul(-1, grad_avg, grad_avg)
                        .add(group["eps"])
                        .sqrt_()
                    )
                else:
                    avg = square_avg.add(group["eps"]).sqrt_()  # eps moved in sqrt

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    # Tensorflow accumulates the LR scaling in the momentum buffer
                    if "lr_in_momentum" in group and group["lr_in_momentum"]:
                        buf.mul_(group["momentum"]).addcdiv_(group["lr"], grad, avg)
                        p.data.add_(-buf)
                    else:
                        # PyTorch scales the param update by LR
                        buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                        p.data.add_(-group["lr"], buf)
                else:
                    p.data.addcdiv_(-group["lr"], grad, avg)

        return loss


@register_optimizer("rmsprop_tf_v2")
class RMSPropTFV2(ClassyOptimizer):
    def __init__(
        self,
        lr: float = 0.01,
        alpha: float = 0.9,
        eps: float = 1e-10,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        decoupled_decay: bool = False,
        lr_in_momentum: bool = True,
    ) -> None:
        super().__init__()

        self.parameters.lr = lr
        self.parameters.alpha = alpha
        self.parameters.eps = eps
        self.parameters.weight_decay = weight_decay
        self.parameters.momentum = momentum
        self.parameters.centered = centered
        self.parameters.decoupled_decay = decoupled_decay
        self.parameters.lr_in_momentum = lr_in_momentum

    def init_pytorch_optimizer(self, model, **kwargs):
        super().init_pytorch_optimizer(model, **kwargs)
        self.optimizer = RMSpropTFV2Optimizer(
            self.param_groups_override,
            lr=self.parameters.lr,
            alpha=self.parameters.alpha,
            eps=self.parameters.eps,
            weight_decay=self.parameters.weight_decay,
            momentum=self.parameters.momentum,
            centered=self.parameters.centered,
            decoupled_decay=self.parameters.decoupled_decay,
            lr_in_momentum=self.parameters.lr_in_momentum,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RMSPropTFV2":
        """Instantiates a RMSPropTFV2 from a configuration.

        Args:
            config: A configuration for a RMSPropTFV2.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A RMSPropTFV2 instance.
        """
        logging.info("Build RMSPropTFV2 optimizer")

        # Default params
        config.setdefault("lr", 0.01)
        config.setdefault("alpha", 0.9)
        config.setdefault("eps", 1e-10)
        config.setdefault("weight_decay", 0.0)
        config.setdefault("momentum", 0.0)
        config.setdefault("centered", False)
        config.setdefault("decoupled_decay", False)
        config.setdefault("lr_in_momentum", True)

        assert is_pos_float(config["lr"])
        for key in ["momentum", "alpha"]:
            assert (
                config[key] >= 0.0 and config[key] < 1.0 and type(config[key]) == float
            ), f"Config must contain a '{key}' in [0, 1) for RMSPropTFV2 optimizer"
        assert is_pos_float(config["eps"])
        assert isinstance(config["centered"], bool)
        assert isinstance(config["decoupled_decay"], bool)
        assert isinstance(config["lr_in_momentum"], bool)

        return cls(
            lr=config["lr"],
            alpha=config["alpha"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
            centered=config["centered"],
            decoupled_decay=config["decoupled_decay"],
            lr_in_momentum=config["lr_in_momentum"],
        )
