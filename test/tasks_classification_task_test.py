#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import shutil
import tempfile
import unittest
from test.generic.config_utils import get_fast_test_task_config, get_test_task_config
from test.generic.utils import (
    LimitedPhaseTrainer,
    compare_model_state,
    compare_samples,
    compare_states,
)

import torch
import torch.nn as nn
from classy_vision.dataset import build_dataset
from classy_vision.generic.distributed_util import is_distributed_training_run
from classy_vision.generic.util import get_checkpoint_dict
from classy_vision.hooks import CheckpointHook, LossLrMeterLoggingHook
from classy_vision.losses import ClassyLoss, build_loss, register_loss
from classy_vision.models import ClassyModel, build_model
from classy_vision.optim import SGD, build_optimizer
from classy_vision.tasks import ClassificationTask, build_task
from classy_vision.trainer import LocalTrainer


@register_loss("test_stateful_loss")
class TestStatefulLoss(ClassyLoss):
    def __init__(self, in_plane):
        super(TestStatefulLoss, self).__init__()

        self.alpha = torch.nn.Parameter(torch.Tensor(in_plane, 2))
        torch.nn.init.xavier_normal(self.alpha)

    @classmethod
    def from_config(cls, config) -> "TestStatefulLoss":
        return cls(in_plane=config["in_plane"])

    def forward(self, output, target):
        value = output.matmul(self.alpha)
        loss = torch.mean(torch.abs(value))

        return loss


# Generate a simple model that has a very high gradient w.r.t. to this
# loss
class SimpleModel(ClassyModel):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(5.0), requires_grad=True)

    def forward(self, x):
        return x + self.param

    @classmethod
    def from_config(cls):
        return cls()


class SimpleLoss(nn.Module):
    def forward(self, x, y):
        return x.pow(2).mean()


class TestClassificationTask(unittest.TestCase):
    def _compare_model_state(self, model_state_1, model_state_2, check_heads=True):
        compare_model_state(self, model_state_1, model_state_2, check_heads)

    def _compare_samples(self, sample_1, sample_2):
        compare_samples(self, sample_1, sample_2)

    def _compare_states(self, state_1, state_2, check_heads=True):
        compare_states(self, state_1, state_2)

    def setUp(self):
        # create a base directory to write checkpoints to
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        # delete all the temporary data created
        shutil.rmtree(self.base_dir)

    def test_build_task(self):
        config = get_test_task_config()
        task = build_task(config)
        self.assertTrue(isinstance(task, ClassificationTask))

    def test_hooks_config_builds_correctly(self):
        config = get_test_task_config()
        config["hooks"] = [{"name": "loss_lr_meter_logging"}]
        task = build_task(config)
        self.assertTrue(len(task.hooks) == 1)
        self.assertTrue(isinstance(task.hooks[0], LossLrMeterLoggingHook))

    def test_get_state(self):
        config = get_test_task_config()
        loss = build_loss(config["loss"])
        task = (
            ClassificationTask()
            .set_num_epochs(1)
            .set_loss(loss)
            .set_model(build_model(config["model"]))
            .set_optimizer(build_optimizer(config["optimizer"]))
        )
        for phase_type in ["train", "test"]:
            dataset = build_dataset(config["dataset"][phase_type])
            task.set_dataset(dataset, phase_type)

        task.prepare()

        task = build_task(config)
        task.prepare()

    def test_synchronize_losses_non_distributed(self):
        """
        Tests that synchronize losses has no side effects in a non-distributed setting.
        """
        test_config = get_fast_test_task_config()
        task = build_task(test_config)
        task.prepare()

        old_losses = copy.deepcopy(task.losses)
        task.synchronize_losses()
        self.assertEqual(old_losses, task.losses)

    def test_synchronize_losses_when_losses_empty(self):
        config = get_fast_test_task_config()
        task = build_task(config)
        task.prepare()

        task.set_use_gpu(torch.cuda.is_available())

        # Losses should be empty when creating task
        self.assertEqual(len(task.losses), 0)

        task.synchronize_losses()

    def test_checkpointing(self):
        """
        Tests checkpointing by running train_steps to make sure the train_steps
        run the same way after loading from a checkpoint.
        """
        config = get_fast_test_task_config()
        task = build_task(config).set_hooks([LossLrMeterLoggingHook()])
        task_2 = build_task(config).set_hooks([LossLrMeterLoggingHook()])

        task.set_use_gpu(torch.cuda.is_available())

        # only train 1 phase at a time
        trainer = LimitedPhaseTrainer(num_phases=1)

        while not task.done_training():
            # set task's state as task_2's checkpoint
            task_2._set_checkpoint_dict(get_checkpoint_dict(task, {}, deep_copy=True))

            # task 2 should have the same state before training
            self._compare_states(task.get_classy_state(), task_2.get_classy_state())

            # train for one phase
            trainer.train(task)
            trainer.train(task_2)

            # task 2 should have the same state after training
            self._compare_states(task.get_classy_state(), task_2.get_classy_state())

    def test_final_train_checkpoint(self):
        """Test that a train phase checkpoint with a where of 1.0 can be loaded"""

        config = get_fast_test_task_config()
        task = build_task(config).set_hooks(
            [CheckpointHook(self.base_dir, {}, phase_types=["train"])]
        )
        task_2 = build_task(config)

        task.set_use_gpu(torch.cuda.is_available())

        trainer = LocalTrainer()
        trainer.train(task)

        self.assertAlmostEqual(task.where, 1.0, delta=1e-3)

        # set task_2's state as task's final train checkpoint
        task_2.set_checkpoint(self.base_dir)
        task_2.prepare()

        # we should be able to train the task
        trainer.train(task_2)

    def test_test_only_checkpointing(self):
        """
        Tests checkpointing by running train_steps to make sure the
        train_steps run the same way after loading from a training
        task checkpoint on a test_only task.
        """
        train_config = get_fast_test_task_config()
        train_config["num_epochs"] = 10
        test_config = get_fast_test_task_config()
        test_config["test_only"] = True
        train_task = build_task(train_config).set_hooks([LossLrMeterLoggingHook()])
        test_only_task = build_task(test_config).set_hooks([LossLrMeterLoggingHook()])

        # prepare the tasks for the right device
        train_task.prepare()

        # test in both train and test mode
        trainer = LocalTrainer()
        trainer.train(train_task)

        # set task's state as task_2's checkpoint
        test_only_task._set_checkpoint_dict(
            get_checkpoint_dict(train_task, {}, deep_copy=True)
        )
        test_only_task.prepare()
        test_state = test_only_task.get_classy_state()

        # We expect the phase idx to be different for a test only task
        self.assertEqual(test_state["phase_idx"], -1)

        # We expect that test only state is test, no matter what train state is
        self.assertFalse(test_state["train"])

        # Num updates should be 0
        self.assertEqual(test_state["num_updates"], 0)

        # train_phase_idx should -1
        self.assertEqual(test_state["train_phase_idx"], -1)

        # Verify task will run
        trainer = LocalTrainer()
        trainer.train(test_only_task)

    def test_test_only_task(self):
        """
        Tests the task in test mode by running train_steps
        to make sure the train_steps run as expected on a
        test_only task
        """
        test_config = get_fast_test_task_config()
        test_config["test_only"] = True

        # delete train dataset
        del test_config["dataset"]["train"]

        test_only_task = build_task(test_config).set_hooks([LossLrMeterLoggingHook()])

        test_only_task.prepare()
        test_state = test_only_task.get_classy_state()

        # We expect that test only state is test, no matter what train state is
        self.assertFalse(test_state["train"])

        # Num updates should be 0
        self.assertEqual(test_state["num_updates"], 0)

        # Verify task will run
        trainer = LocalTrainer()
        trainer.train(test_only_task)

    def test_train_only_task(self):
        """
        Tests that the task runs when only a train dataset is specified.
        """
        test_config = get_fast_test_task_config()

        # delete the test dataset from the config
        del test_config["dataset"]["test"]

        task = build_task(test_config).set_hooks([LossLrMeterLoggingHook()])
        task.prepare()

        # verify the the task can still be trained
        trainer = LocalTrainer()
        trainer.train(task)

    @unittest.skipUnless(torch.cuda.is_available(), "This test needs a gpu to run")
    def test_checkpointing_different_device(self):
        config = get_fast_test_task_config()
        task = build_task(config)
        task_2 = build_task(config)

        for use_gpu in [True, False]:
            task.set_use_gpu(use_gpu)
            task.prepare()

            # set task's state as task_2's checkpoint
            task_2._set_checkpoint_dict(get_checkpoint_dict(task, {}, deep_copy=True))

            # we should be able to run the trainer using state from a different device
            trainer = LocalTrainer()
            task_2.set_use_gpu(not use_gpu)
            trainer.train(task_2)

    @unittest.skipUnless(
        is_distributed_training_run(), "This test needs a distributed run"
    )
    def test_get_classy_state_on_loss(self):
        config = get_fast_test_task_config()
        config["loss"] = {"name": "test_stateful_loss", "in_plane": 256}
        task = build_task(config)
        task.prepare()
        self.assertIn("alpha", task.get_classy_state()["loss"])

    def test_gradient_clipping(self):
        apex_available = True
        try:
            import apex  # noqa F401
        except ImportError:
            apex_available = False

        def train_with_clipped_gradients(amp_args=None):
            task = build_task(get_fast_test_task_config())
            task.set_num_epochs(1)
            task.set_model(SimpleModel())
            task.set_loss(SimpleLoss())
            task.set_meters([])
            task.set_use_gpu(torch.cuda.is_available())
            task.set_clip_grad_norm(0.5)
            task.set_amp_args(amp_args)

            task.set_optimizer(SGD(lr=1))

            trainer = LocalTrainer()
            trainer.train(task)

            return task.model.param.grad.norm()

        grad_norm = train_with_clipped_gradients(None)
        self.assertAlmostEqual(grad_norm, 0.5, delta=1e-2)

        if apex_available and torch.cuda.is_available():
            grad_norm = train_with_clipped_gradients({"opt_level": "O2"})
            self.assertAlmostEqual(grad_norm, 0.5, delta=1e-2)

    def test_clip_stateful_loss(self):
        config = get_fast_test_task_config()
        config["loss"] = {"name": "test_stateful_loss", "in_plane": 256}
        config["grad_norm_clip"] = grad_norm_clip = 1
        task = build_task(config)
        task.set_use_gpu(False)
        task.prepare()

        # set fake gradients with norm > grad_norm_clip
        for param in itertools.chain(
            task.base_model.parameters(), task.base_loss.parameters()
        ):
            param.grad = 1.1 + torch.rand(param.shape)
            self.assertGreater(param.grad.norm(), grad_norm_clip)

        task._clip_gradients(grad_norm_clip)

        for param in itertools.chain(
            task.base_model.parameters(), task.base_loss.parameters()
        ):
            self.assertLessEqual(param.grad.norm(), grad_norm_clip)

    # helper used by gradient accumulation tests
    def train_with_batch(self, simulated_bs, actual_bs, clip_grad_norm=None):
        config = copy.deepcopy(get_fast_test_task_config())
        config["dataset"]["train"]["num_samples"] = 12
        config["dataset"]["train"]["batchsize_per_replica"] = actual_bs
        del config["dataset"]["test"]

        task = build_task(config)
        task.set_num_epochs(1)
        task.set_model(SimpleModel())
        task.set_loss(SimpleLoss())
        task.set_meters([])
        task.set_use_gpu(torch.cuda.is_available())
        if simulated_bs is not None:
            task.set_simulated_global_batchsize(simulated_bs)
        if clip_grad_norm is not None:
            task.set_clip_grad_norm(clip_grad_norm)

        task.set_optimizer(SGD(lr=1))

        trainer = LocalTrainer()
        trainer.train(task)

        return task.model.param

    def test_gradient_accumulation(self):
        param_with_accumulation = self.train_with_batch(simulated_bs=4, actual_bs=2)
        param = self.train_with_batch(simulated_bs=4, actual_bs=4)

        self.assertAlmostEqual(param_with_accumulation, param, delta=1e-5)

    def test_gradient_accumulation_and_clipping(self):
        param = self.train_with_batch(simulated_bs=6, actual_bs=2, clip_grad_norm=0.1)

        # param starts at 5, it has to decrease, LR = 1
        # clipping the grad to 0.1 means we drop 0.1 per update. num_samples =
        # 12 and the simulated batch size is 6, so we should do 2 updates: 5 ->
        # 4.9 -> 4.8
        self.assertAlmostEqual(param, 4.8, delta=1e-5)
