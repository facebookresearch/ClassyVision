from typing import Any, Dict

import torch.optim

from . import ClassyOptimizer, register_optimizer


@register_optimizer("adadelta")
class Adadelta(ClassyOptimizer):
    def __init__(
        self,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0
    ) -> None:
        super.__init__()

        self._lr = lr
        self._rho = rho
        self._eps = eps
        self._weight_decay = weight_decay

    def prepare(self, param_groups) -> None:

        self.optimizer = torch.optim.Adadelta(
            param_groups,
            lr=self._lr,
            rho=self._rho,
            eps=self._eps,
            weight_decay=self._weight_decay
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Adadelta":
        """Instantiates a Adadelta from a configuration.

        Args:
            config: A configuration for a Adadelta.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            An Adadelta instance.
        """
        # Default params
        config.setdefault("lr", 1.0)
        config.setdefault("rho", 0.9)
        config.setdefault("eps", 1e-6)
        config.setdefault("weight_decay", 0)


        return cls(
            lr=config["lr"],
            rho=config["rho"],
            eps=config["eps"],
            weight_decay=["weight_decay"],
        )
