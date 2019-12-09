#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import torch
from classy_vision.models import ClassyModel
from classy_vision.optim import build_optimizer


class TestOptimizer(ABC):
    @abstractmethod
    def _get_config(self):
        pass

    @abstractmethod
    def _instance_to_test(self):
        pass

    def _check_momentum_buffer(self):
        return False

    def _get_optimizer_params(self):
        return {
            "regularized_params": [
                torch.tensor([[1.0, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 4.0]], requires_grad=True),
            ],
            "unregularized_params": [torch.tensor([[1.0, 2.0]], requires_grad=True)],
        }

    def _get_mock_classy_vision_model(self, trainable_params=True):
        mock_classy_vision_model = ClassyModel()

        if trainable_params:
            mock_classy_vision_model.get_optimizer_params = MagicMock(
                return_value=self._get_optimizer_params()
            )
            mock_classy_vision_model.parameters = MagicMock(
                return_value=self._get_optimizer_params()["regularized_params"]
                + self._get_optimizer_params()["unregularized_params"]
            )
        else:
            mock_classy_vision_model.get_optimizer_params = MagicMock(
                return_value={"regularized_params": [], "unregularized_params": []}
            )
            mock_classy_vision_model.parameters = MagicMock(
                return_value=[
                    param.detach()
                    for param in self._get_optimizer_params()["regularized_params"]
                    + self._get_optimizer_params()["unregularized_params"]
                ]
            )

        return mock_classy_vision_model

    def _set_gradient(self, params, grad_values=None):
        if grad_values is None:
            grad_values = [0.1, 0.1]
        for i in range(len(params)):
            params[i].grad = torch.tensor([grad_values])

    def _set_model_gradient(self, model, grad_values=None):
        for param_type in ["regularized_params", "unregularized_params"]:
            self._set_gradient(model.get_optimizer_params()[param_type], grad_values)

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

        mock_classy_vision_model = self._get_mock_classy_vision_model()
        opt1 = build_optimizer(config)
        opt1.init_pytorch_optimizer(mock_classy_vision_model)

        self._set_model_gradient(mock_classy_vision_model, grad_values)
        opt1.step()
        state = opt1.get_classy_state()

        config["lr"] += 0.1
        opt2 = build_optimizer(config)
        opt2.init_pytorch_optimizer(mock_classy_vision_model)
        self.assertTrue(isinstance(opt1, self._instance_to_test()))
        opt2.set_classy_state(state)
        self.assertEqual(opt1.parameters, opt2.parameters)
        for i in range(len(opt1.optimizer.param_groups[0]["params"])):
            self.assertTrue(
                torch.allclose(
                    opt1.optimizer.param_groups[0]["params"][i],
                    opt2.optimizer.param_groups[0]["params"][i],
                )
            )
        self._compare_momentum_values(
            opt1.get_classy_state()["optim"], opt2.get_classy_state()["optim"]
        )

        # check if the optimizers behave the same on params update
        mock_classy_vision_model1 = self._get_mock_classy_vision_model()
        mock_classy_vision_model2 = self._get_mock_classy_vision_model()
        self._set_model_gradient(mock_classy_vision_model1, grad_values)
        self._set_model_gradient(mock_classy_vision_model2, grad_values)
        opt1 = build_optimizer(config)
        opt1.init_pytorch_optimizer(mock_classy_vision_model1)
        opt2 = build_optimizer(config)
        opt2.init_pytorch_optimizer(mock_classy_vision_model2)
        opt1.step()
        opt2.step()
        for i in range(len(opt1.optimizer.param_groups[0]["params"])):
            print(opt1.optimizer.param_groups[0]["params"][i])
            self.assertTrue(
                torch.allclose(
                    opt1.optimizer.param_groups[0]["params"][i],
                    opt2.optimizer.param_groups[0]["params"][i],
                )
            )
        self._compare_momentum_values(
            opt1.get_classy_state()["optim"], opt2.get_classy_state()["optim"]
        )

    def test_build_sgd(self):
        config = self._get_config()
        mock_classy_vision_model = self._get_mock_classy_vision_model(
            trainable_params=True
        )
        opt = build_optimizer(config)
        opt.init_pytorch_optimizer(mock_classy_vision_model)
        self.assertTrue(isinstance(opt, self._instance_to_test()))

    def test_raise_error_on_non_trainable_params(self):
        # Test Raise ValueError if there are no trainable params in the model.
        config = self._get_config()
        with self.assertRaises(ValueError):
            opt = build_optimizer(config)
            opt.init_pytorch_optimizer(
                self._get_mock_classy_vision_model(trainable_params=False)
            )

    def test_get_set_state(self):
        for grad_values in [[0.1, 0.1], [-0.1, -0.1], [0.0, 0.0], [0.1, -0.1]]:
            self._get_set_state(grad_values)

    def test_set_invalid_state(self):
        config = self._get_config()
        mock_classy_vision_model = self._get_mock_classy_vision_model()
        opt = build_optimizer(config)
        opt.init_pytorch_optimizer(mock_classy_vision_model)
        self.assertTrue(isinstance(opt, self._instance_to_test()))

        with self.assertRaises(KeyError):
            opt.set_classy_state({})

    def test_lr_schedule(self):
        config = self._get_config()

        mock_classy_vision_model = self._get_mock_classy_vision_model()
        opt = build_optimizer(config)
        opt.init_pytorch_optimizer(mock_classy_vision_model)

        # Test initial learning rate
        for group in opt.optimizer.param_groups:
            self.assertEqual(group["lr"], 0.1)

        def _test_lr_schedule(optimizer, num_epochs, epochs, targets):
            for i in range(len(epochs)):
                epoch = epochs[i]
                target = targets[i]
                param_groups = optimizer.optimizer.param_groups.copy()
                optimizer.update_schedule_on_epoch(epoch / num_epochs)
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
        config["lr"] = {"name": "step", "values": [0.1, 0.01, 0.001]}
        opt = build_optimizer(config)
        opt.init_pytorch_optimizer(mock_classy_vision_model)
        targets = [0.1] * 8 + [0.01] * 3 + [0.001] * 4
        _test_lr_schedule(opt, num_epochs, epochs, targets)

        # Test step learning schedule with warmup
        init_lr = 0.01
        warmup_epochs = 0.1
        config["lr"] = {
            "name": "composite",
            "schedulers": [
                {"name": "linear", "start_lr": init_lr, "end_lr": 0.1},
                {"name": "step", "values": [0.1, 0.01, 0.001]},
            ],
            "update_interval": "epoch",
            "interval_scaling": ["rescaled", "fixed"],
            "lengths": [warmup_epochs / num_epochs, 1 - warmup_epochs / num_epochs],
        }

        opt = build_optimizer(config)
        opt.init_pytorch_optimizer(mock_classy_vision_model)
        targets = [0.01, 0.0325, 0.055] + [0.1] * 5 + [0.01] * 3 + [0.001] * 4
        _test_lr_schedule(opt, num_epochs, epochs, targets)
