from typing import Any, Dict

import torch.optim

from . import ClassyOptimizer, register_optimizer


@register_optimzer("adadelta")
class Adadelta(ClassyOptimizer):
    def __init__(
        self,
        lr: float = 0.1,
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
        """Instantiates a Adam from a configuration.

        Args:
            config: A configuration for a Adam.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            An Adam instance.
        """
        # Default params
        config.setdefault("lr", 0.1)
        config.setdefalt("rho", 0.9)
        config.setdefult("eps", 1e-6)
        config.setdefult("weight_decay", 0)


        return cls(
            lr=config["lr"]
            rho=config["rho"]
            eps=config["eps"]
            weight_decay=["weight_decay"]
        )
