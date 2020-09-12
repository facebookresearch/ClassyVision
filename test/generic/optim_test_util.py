#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from classy_vision.generic.util import split_batchnorm_params
from classy_vision.models import ClassyModel
from classy_vision.optim import build_optimizer, build_optimizer_schedulers
from classy_vision.optim.param_scheduler import LinearParamScheduler


class TestOptimizer(ABC):
    @abstractmethod
    def _get_config(self):
        pass

    @abstractmethod
    def _instance_to_test(self):
        pass

    def _check_momentum_buffer(self):
        return False

    def _parameters(self, requires_grad=True):
        return [
            torch.tensor([[1.0, 2.0]], requires_grad=requires_grad),
            torch.tensor([[3.0, 4.0]], requires_grad=requires_grad),
        ]

    def _set_gradient(self, params, grad_values=None):
        if grad_values is None:
            grad_values = [0.1, 0.1]
        for i in range(len(params)):
            params[i].grad = torch.tensor([grad_values])

    def _compare_momentum_values(self, optim1, optim2):
        self.assertEqual(len(optim1["param_groups"]), len(optim2["param_groups"]))

        for i in range(len(optim1["param_groups"])):
            self.assertEqual(
                len(optim1["param_groups"][i]["params"]),
                len(optim2["param_groups"][i]["params"]),
            )
            if self._check_momentum_buffer():
                for j in range(len(optim1["param_groups"][i]["params"])):
                    id1 = optim1["param_groups"][i]["params"][j]
                    id2 = optim2["param_groups"][i]["params"][j]
                    self.assertTrue(
                        torch.allclose(
                            optim1["state"][id1]["momentum_buffer"],
                            optim2["state"][id2]["momentum_buffer"],
                        )
                    )

    def _get_set_state(self, grad_values):
        config = self._get_config()

        opt1 = build_optimizer(config)
        opt1.set_param_groups(self._parameters(), lr=1, momentum=0.9)
        self.assertIsInstance(opt1, self._instance_to_test())

        self._set_gradient(self._parameters(), grad_values)
        opt1.step(where=0)

        if config["name"] == "zero":
            opt1.consolidate_state_dict()

        state = opt1.get_classy_state()

        opt2 = build_optimizer(config)
        opt2.set_param_groups(self._parameters(), lr=2)

        self.assertNotEqual(opt1.options_view.lr, opt2.options_view.lr)
        opt2.set_classy_state(state)
        self.assertEqual(opt1.options_view.lr, opt2.options_view.lr)

        for i in range(len(opt1.optimizer.param_groups[0]["params"])):
            self.assertTrue(
                torch.allclose(
                    opt1.optimizer.param_groups[0]["params"][i],
                    opt2.optimizer.param_groups[0]["params"][i],
                )
            )

        if config["name"] == "zero":
            opt2.consolidate_state_dict()

        self._compare_momentum_values(
            opt1.get_classy_state()["optim"], opt2.get_classy_state()["optim"]
        )

        # check if the optimizers behave the same on params update
        mock_classy_vision_model1 = self._parameters()
        mock_classy_vision_model2 = self._parameters()
        self._set_gradient(mock_classy_vision_model1, grad_values)
        self._set_gradient(mock_classy_vision_model2, grad_values)
        opt1 = build_optimizer(config)
        opt1.set_param_groups(mock_classy_vision_model1)
        opt2 = build_optimizer(config)
        opt2.set_param_groups(mock_classy_vision_model2)
        opt1.step(where=0)
        opt2.step(where=0)
        for i in range(len(opt1.optimizer.param_groups[0]["params"])):
            print(opt1.optimizer.param_groups[0]["params"][i])
            self.assertTrue(
                torch.allclose(
                    opt1.optimizer.param_groups[0]["params"][i],
                    opt2.optimizer.param_groups[0]["params"][i],
                )
            )

        if config["name"] == "zero":
            opt1.consolidate_state_dict()
            opt2.consolidate_state_dict()

        self._compare_momentum_values(
            opt1.get_classy_state()["optim"], opt2.get_classy_state()["optim"]
        )

    def test_build_sgd(self):
        config = self._get_config()
        opt = build_optimizer(config)
        opt.set_param_groups(self._parameters())
        self.assertTrue(isinstance(opt, self._instance_to_test()))

    def test_get_set_state(self):
        for grad_values in [[0.1, 0.1], [-0.1, -0.1], [0.0, 0.0], [0.1, -0.1]]:
            self._get_set_state(grad_values)

    def test_set_invalid_state(self):
        config = self._get_config()
        opt = build_optimizer(config)
        opt.set_param_groups(self._parameters())
        self.assertTrue(isinstance(opt, self._instance_to_test()))

        with self.assertRaises(KeyError):
            opt.set_classy_state({})

    def test_lr_schedule(self):
        config = self._get_config()

        opt = build_optimizer(config)
        param_schedulers = build_optimizer_schedulers(config)
        opt.set_param_groups({"params": self._parameters(), **param_schedulers})

        # Test initial learning rate
        for group in opt.optimizer.param_groups:
            self.assertEqual(group["lr"], 0.1)

        def _test_lr_schedule(optimizer, num_epochs, epochs, targets):
            for i in range(len(epochs)):
                epoch = epochs[i]
                target = targets[i]
                param_groups = optimizer.optimizer.param_groups.copy()
                optimizer.on_epoch(epoch / num_epochs)
                for idx, group in enumerate(optimizer.optimizer.param_groups):
                    self.assertEqual(group["lr"], target)
                    # Make sure all but LR is same
                    param_groups[idx]["lr"] = target
                    self.assertEqual(param_groups[idx], group)

        # Test constant learning schedule
        num_epochs = 90
        epochs = [0, 0.025, 0.05, 0.1, 0.5, 1, 15, 29, 30, 31, 59, 60, 61, 88, 89]
        targets = [0.1] * 15
        _test_lr_schedule(opt, num_epochs, epochs, targets)

        # Test step learning schedule
        config["param_schedulers"] = {
            "lr": {"name": "step", "values": [0.1, 0.01, 0.001]}
        }
        opt = build_optimizer(config)
        param_schedulers = build_optimizer_schedulers(config)
        opt.set_param_groups({"params": self._parameters(), **param_schedulers})

        targets = [0.1] * 8 + [0.01] * 3 + [0.001] * 4
        _test_lr_schedule(opt, num_epochs, epochs, targets)

        # Test step learning schedule with warmup
        init_lr = 0.01
        warmup_epochs = 0.1
        config["param_schedulers"] = {
            "lr": {
                "name": "composite",
                "schedulers": [
                    {"name": "linear", "start_value": init_lr, "end_value": 0.1},
                    {"name": "step", "values": [0.1, 0.01, 0.001]},
                ],
                "update_interval": "epoch",
                "interval_scaling": ["rescaled", "fixed"],
                "lengths": [warmup_epochs / num_epochs, 1 - warmup_epochs / num_epochs],
            }
        }

        opt = build_optimizer(config)
        param_schedulers = build_optimizer_schedulers(config)
        opt.set_param_groups({"params": self._parameters(), **param_schedulers})

        targets = [0.01, 0.0325, 0.055] + [0.1] * 5 + [0.01] * 3 + [0.001] * 4
        _test_lr_schedule(opt, num_epochs, epochs, targets)

    def test_set_param_groups(self):
        opt = build_optimizer(self._get_config())
        # This must crash since we're missing the .set_param_groups call
        with self.assertRaises(RuntimeError):
            opt.step(where=0)

    def test_step_args(self):
        opt = build_optimizer(self._get_config())
        opt.set_param_groups([torch.tensor([1.0], requires_grad=True)])

        # where argument must be named explicitly
        with self.assertRaises(RuntimeError):
            opt.step(0)

        # this shouldn't crash
        opt.step(where=0)

    def test_get_lr(self):
        opt = build_optimizer(self._get_config())
        param = torch.tensor([1.0], requires_grad=True)
        opt.set_param_groups([{"params": [param], "lr": 1}])

        self.assertEqual(opt.options_view.lr, 1)

        # Case two: verify LR changes
        opt = build_optimizer(self._get_config())
        param = torch.tensor([1.0], requires_grad=True)
        opt.set_param_groups([{"params": [param], "lr": LinearParamScheduler(1, 2)}])

        self.assertAlmostEqual(opt.options_view.lr, 1)
        opt.step(where=0.5)
        self.assertAlmostEqual(opt.options_view.lr, 1.5)

    def test_batchnorm_weight_decay(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 3)
                self.relu = nn.ReLU()
                self.bn = nn.BatchNorm1d(3)

            def forward(self, x):
                return self.bn(self.relu(self.lin(x)))

        torch.manual_seed(1)
        model = MyModel()

        opt = build_optimizer(self._get_config())
        bn_params, lin_params = split_batchnorm_params(model)

        lin_param_before = model.lin.weight.detach().clone()
        bn_param_before = model.bn.weight.detach().clone()

        with torch.enable_grad():
            x = torch.tensor([[1.0, 1.0], [1.0, 2.0]])
            out = model(x).pow(2).sum()
            out.backward()

        opt.set_param_groups(
            [
                {
                    "params": lin_params,
                    "lr": LinearParamScheduler(1, 2),
                    "weight_decay": 0.5,
                },
                {"params": bn_params, "lr": 0, "weight_decay": 0},
            ]
        )

        opt.step(where=0.5)

        # Make sure the linear parameters are trained but not the batch norm
        self.assertFalse(torch.allclose(model.lin.weight, lin_param_before))
        self.assertTrue(torch.allclose(model.bn.weight, bn_param_before))

        opt.step(where=0.5)

        # Same, but after another step and triggering the lr scheduler
        self.assertFalse(torch.allclose(model.lin.weight, lin_param_before))
        self.assertTrue(torch.allclose(model.bn.weight, bn_param_before))
