#!/usr/bin/env python3

from torch.nn.modules.loss import _WeightedLoss


class ClassyCriterion(_WeightedLoss):
    def __init__(self, config):
        """
        Classy Criterion constructor. This stores the criterion config for
        future access and constructs the basic criterion object.
        """
        super(ClassyCriterion, self).__init__()
        self._config = config

    def forward(self, output, target):
        """
        Compute the loss for the provided sample.
        """
        raise NotImplementedError
