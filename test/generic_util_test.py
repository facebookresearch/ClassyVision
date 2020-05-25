#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import shutil
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from test.generic.config_utils import get_fast_test_task_config, get_test_model_configs
from test.generic.utils import compare_model_state, compare_states

import classy_vision.generic.util as util
import torch
import torch.nn as nn
from classy_vision.generic.util import (
    CHECKPOINT_FILE,
    load_checkpoint,
    save_checkpoint,
    update_classy_model,
    update_classy_state,
)
from classy_vision.models import build_model
from classy_vision.tasks import build_task
from classy_vision.trainer import LocalTrainer


ROOT = Path(__file__).parent


@mock.patch("torch.tensor")
def get_mock_tensor(mock_class):
    def get_cuda_tensor():
        t = torch.tensor([1, 2, 3])
        t.is_cuda = True
        return t

    mock_class.return_value.cuda.return_value = get_cuda_tensor()
    mock_class.is_cuda = False
    return torch.tensor([1, 2, 3])


class TestUtilMethods(unittest.TestCase):
    def test_recursive_copy_to_gpu(self):
        tensor_a = get_mock_tensor()
        tensor_b = get_mock_tensor()

        valid_gpu_copy_value = tensor_a
        gpu_value = util.recursive_copy_to_gpu(valid_gpu_copy_value)
        self.assertTrue(gpu_value.is_cuda)

        valid_recursive_copy_value = [[tensor_a]]
        gpu_value = util.recursive_copy_to_gpu(valid_recursive_copy_value)
        self.assertTrue(gpu_value[0][0].is_cuda)

        valid_gpu_copy_collections = [
            (tensor_a, tensor_b),
            [tensor_a, tensor_b],
            {"tensor_a": tensor_a, "tensor_b": tensor_b},
        ]
        for value in valid_gpu_copy_collections:
            gpu_value = util.recursive_copy_to_gpu(value)
            if isinstance(value, dict):
                self.assertTrue(gpu_value["tensor_a"].is_cuda)
                self.assertTrue(gpu_value["tensor_b"].is_cuda)
            else:
                self.assertEqual(len(gpu_value), 2)
                self.assertTrue(gpu_value[0].is_cuda)
                self.assertTrue(gpu_value[1].is_cuda)

        invalid_gpu_copy_depth = [
            ((((tensor_a, tensor_b), tensor_b), tensor_b), tensor_b),
            {"tensor_map_a": {"tensor_map_b": {"tensor_map_c": {"tensor": tensor_a}}}},
            [[[[tensor_a, tensor_b], tensor_b], tensor_b], tensor_b],
        ]
        for value in invalid_gpu_copy_depth:
            with self.assertRaises(ValueError):
                gpu_value = util.recursive_copy_to_gpu(value, max_depth=3)

        value = {"a": "b"}
        self.assertEqual(value, util.recursive_copy_to_gpu(value))

    _json_config_file = ROOT / "generic_util_json_blob_test.json"

    def _get_config(self):
        return {
            "name": "test_task",
            "num_epochs": 12,
            "loss": {"name": "test_loss"},
            "dataset": {
                "name": "test_data",
                "batchsize_per_replica": 8,
                "use_pairs": False,
                "num_samples": None,
                "use_shuffle": {"train": True, "test": False},
            },
            "meters": [{"name": "test_meter", "test_param": 0.1}],
            "model": {"name": "test_model", "architecture": [1, 2, 3, 4]},
            "optimizer": {
                "name": "test_optimizer",
                "test_param": {
                    "name": "test_scheduler",
                    "values": [0.1, 0.01, 0.001, 0.0001],
                },
            },
        }

    def test_load_config(self):
        expected_config = self._get_config()
        config = util.load_json(self._json_config_file)

        self.assertEqual(config, expected_config)

    def test_torch_seed(self):
        # test that using util.torch_seed doesn't impact the generation of
        # random numbers outside its context and that random numbers generated
        # within its context are the same as setting a manual seed
        torch.manual_seed(0)
        torch.randn(10)
        random_tensor_1 = torch.randn(10)
        torch.manual_seed(0)
        torch.randn(10)
        with util.torch_seed(1):
            random_tensor_2 = torch.randn(10)
        self.assertTrue(torch.equal(torch.randn(10), random_tensor_1))
        torch.manual_seed(1)
        self.assertTrue(torch.equal(torch.randn(10), random_tensor_2))

    def test_get_model_dummy_input(self):
        for config in get_test_model_configs():
            model = build_model(config)  # pass in a dummy model for the cuda check
            batchsize = 8
            # input_key is list
            input_key = ["audio", "video"]
            input_shape = [[3, 40, 100], [4, 16, 223, 223]]  # dummy input shapes
            result = util.get_model_dummy_input(
                model, input_shape, input_key, batchsize
            )
            self.assertEqual(result.keys(), {"audio", "video"})
            for i in range(len(input_key)):
                self.assertEqual(
                    result[input_key[i]].size(), tuple([batchsize] + input_shape[i])
                )
            # input_key is string
            input_key = "video"
            input_shape = [4, 16, 223, 223]
            result = util.get_model_dummy_input(
                model, input_shape, input_key, batchsize
            )
            self.assertEqual(result.keys(), {"video"})
            self.assertEqual(result[input_key].size(), tuple([batchsize] + input_shape))
            # input_key is None
            input_key = None
            input_shape = [4, 16, 223, 223]
            result = util.get_model_dummy_input(
                model, input_shape, input_key, batchsize
            )
            self.assertEqual(result.size(), tuple([batchsize] + input_shape))

    def _compare_model_train_mode(self, model_1, model_2):
        for name_1, module_1 in model_1.named_modules():
            found = False
            for name_2, module_2 in model_2.named_modules():
                if name_1 == name_2:
                    found = True
                    if module_1.training != module_2.training:
                        return False
            if not found:
                return False
        return True

    def _check_model_train_mode(self, model, expected_mode):
        for module in model.modules():
            if module.training != expected_mode:
                return False
        return True

    def test_train_model_eval_model(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 2)
                self.dropout = nn.Dropout()
                self.seq = nn.Sequential(
                    nn.ReLU(), nn.Conv2d(1, 2, 3), nn.BatchNorm2d(1, 2)
                )

        test_model = TestModel()
        for train in [True, False]:
            test_model.train(train)

            # flip some of the modes
            test_model.dropout.train(not train)
            test_model.seq[1].train(not train)

            orig_model = copy.deepcopy(test_model)

            with util.train_model(test_model):
                self._check_model_train_mode(test_model, True)
                # the modes should be different inside the context manager
                self.assertFalse(self._compare_model_train_mode(orig_model, test_model))
            self.assertTrue(self._compare_model_train_mode(orig_model, test_model))

            with util.eval_model(test_model):
                self._check_model_train_mode(test_model, False)
                # the modes should be different inside the context manager
                self.assertFalse(self._compare_model_train_mode(orig_model, test_model))
            self.assertTrue(self._compare_model_train_mode(orig_model, test_model))


class TestUpdateStateFunctions(unittest.TestCase):
    def _compare_states(self, state_1, state_2, check_heads=True):
        compare_states(self, state_1, state_2)

    def _compare_model_state(self, state_1, state_2, check_heads=True):
        return compare_model_state(self, state_1, state_2, check_heads=check_heads)

    def test_update_classy_state(self):
        """
        Tests that the update_classy_state successfully updates from a
        checkpoint
        """
        config = get_fast_test_task_config()
        task = build_task(config)
        task_2 = build_task(config)
        task_2.prepare()
        trainer = LocalTrainer()
        trainer.train(task)
        update_classy_state(task_2, task.get_classy_state(deep_copy=True))
        self._compare_states(task.get_classy_state(), task_2.get_classy_state())

    def test_update_classy_model(self):
        """
        Tests that the update_classy_model successfully updates from a
        checkpoint
        """
        config = get_fast_test_task_config()
        task = build_task(config)
        trainer = LocalTrainer()
        trainer.train(task)
        for reset_heads in [False, True]:
            task_2 = build_task(config)
            # prepare task_2 for the right device
            task_2.prepare()
            update_classy_model(
                task_2.model, task.model.get_classy_state(deep_copy=True), reset_heads
            )
            self._compare_model_state(
                task.model.get_classy_state(),
                task_2.model.get_classy_state(),
                check_heads=not reset_heads,
            )
            if reset_heads:
                # the model head states should be different
                with self.assertRaises(Exception):
                    self._compare_model_state(
                        task.model.get_classy_state(),
                        task_2.model.get_classy_state(),
                        check_heads=True,
                    )


class TestCheckpointFunctions(unittest.TestCase):
    def setUp(self):
        # create a base directory to write checkpoints to
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        # delete all the temporary data created
        shutil.rmtree(self.base_dir)

    def test_save_and_load_checkpoint(self):
        checkpoint_dict = {str(i): i * 2 for i in range(1000)}

        # save to the default checkpoint file
        save_checkpoint(self.base_dir, checkpoint_dict)

        # load the checkpoint by using the default file
        loaded_checkpoint = load_checkpoint(self.base_dir)
        self.assertDictEqual(checkpoint_dict, loaded_checkpoint)

        # load the checkpoint by passing the full path
        checkpoint_path = f"{self.base_dir}/{CHECKPOINT_FILE}"
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        self.assertDictEqual(checkpoint_dict, loaded_checkpoint)

        # create a new checkpoint dict
        filename = "my_checkpoint.torch"
        checkpoint_dict = {str(i): i * 3 for i in range(1000)}

        # save the checkpoint to a different file
        save_checkpoint(self.base_dir, checkpoint_dict, checkpoint_file=filename)

        # load the checkpoint by passing the full path
        checkpoint_path = f"{self.base_dir}/{filename}"
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        self.assertDictEqual(checkpoint_dict, loaded_checkpoint)
