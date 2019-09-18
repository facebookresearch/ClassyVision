#!/usr/bin/env python3

from . import ClassyCriterion, build_criterion, register_criterion


@register_criterion("sum_arbitrary")
class SumArbitraryLoss(ClassyCriterion):
    """
    Sums a collection of (weighted) torch.nn losses.
    """

    def __init__(self, config):
        super(SumArbitraryLoss, self).__init__(config)
        # assertions:
        assert (
            type(config["losses"]) == list and len(config["losses"]) > 0
        ), "losses must be a list of registered losses with length > 0"
        if config["weights"] is None:
            config["weights"] = [1.0] * len(config["losses"])
        assert type(config["weights"]) == list and len(config["weights"]) == len(
            config["losses"]
        ), "weights must be None or a list and have same length as losses"

        loss_modules = []
        for loss_config in config["losses"]:
            loss_modules.append(build_criterion(loss_config))

        assert all(
            isinstance(loss_module, ClassyCriterion) for loss_module in loss_modules
        ), "All losses must be registered, valid ClassyCriterions"

        # create class:
        self.losses = loss_modules
        self.weights = config["weights"]

    def forward(self, prediction, target):
        for idx, loss in enumerate(self.losses):
            current_loss = loss(prediction, target)
            if idx == 0:
                total_loss = current_loss
            else:
                total_loss = total_loss.add(self.weights[idx], current_loss)
        return total_loss
