#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from test.generic.config_utils import get_fast_test_task_config, get_test_model_configs
from test.generic.utils import compare_model_state, compare_states

import classy_vision.generic.util as util
import torch
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
    def _get_base_pred_probs(self):
        return torch.tensor(
            [
                [0.92, 0.08],  # 1
                [0.91, 0.09],  # 0
                [0.89, 0.11],  # 0
                [0.79, 0.21],  # 0
                [0.78, 0.22],  # 0
                [0.69, 0.31],  # 1
                [0.68, 0.32],  # 0
                [0.59, 0.41],  # 1
                [0.58, 0.42],  # 0
                [0.49, 0.51],  # 1
                [0.48, 0.52],  # 1
                [0.39, 0.61],  # 0
                [0.38, 0.62],  # 0
                [0.29, 0.71],  # 1
                [0.28, 0.72],  # 1
                [0.19, 0.81],  # 0
                [0.18, 0.82],  # 1
                [0.09, 0.91],  # 1
                [0.08, 0.92],  # 1
                [0.07, 0.93],  # 0
                [0.06, 0.94],  # 1
                [0.03, 0.97],  # 0
            ]
        )

    def _get_base_targets(self):
        return torch.tensor(
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0]
        ).int()

    def _get_base_class_hist(self):
        return torch.stack(
            [
                torch.Tensor(
                    [1, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 1, 0, 2, 0, 1, 1, 0]
                ),
                torch.Tensor(
                    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 3, 0]
                ),
            ],
            dim=1,
        ).long()

    def _get_base_total_hist(self):
        return torch.stack(
            [
                torch.Tensor(
                    [1, 4, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2, 0]
                ),
                torch.Tensor(
                    [0, 2, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 4, 1]
                ),
            ],
            dim=1,
        ).long()

    def test_create_class_histograms_success(self):
        pred_probs = self._get_base_pred_probs()
        targets = self._get_base_targets()

        class_hist, total_hist = util.create_class_histograms(pred_probs, targets, 20)
        torch.testing.assert_allclose(class_hist, self._get_base_class_hist())
        torch.testing.assert_allclose(total_hist, self._get_base_total_hist())

    def test_create_class_histograms_fail(self):
        pred_probs = self._get_base_pred_probs()
        targets = self._get_base_targets()

        # Torch tensors only
        with self.assertRaises(AssertionError):
            class_hist, total_hist = util.create_class_histograms(
                pred_probs.numpy(), targets, 20
            )

        # Torch tensors only
        with self.assertRaises(AssertionError):
            class_hist, total_hist = util.create_class_histograms(
                pred_probs, targets.numpy(), 20
            )

        # Prediction and target are same size
        with self.assertRaises(AssertionError):
            class_hist, total_hist = util.create_class_histograms(
                pred_probs[0:5, :], targets, 20
            )

        # Prediction is between 0 and 1
        with self.assertRaises(AssertionError):
            pred_probs[0, :] = torch.tensor([-0.1, 1.1])
            class_hist, total_hist = util.create_class_histograms(
                pred_probs, targets, 20
            )

    def test_compute_pr_curves(self):
        class_hist = self._get_base_class_hist()
        total_hist = self._get_base_total_hist()

        pr_curves = util.compute_pr_curves(class_hist, total_hist)
        # For curves without duplicates removed / precisions cleaned
        # up, see: P60302268
        exp_pos_prec = torch.tensor(
            [
                3.0 / 5.0,
                4.0 / 7.0,
                6.0 / 9.0,
                6.0 / 11.0,
                8.0 / 13.0,
                9.0 / 15.0,
                10.0 / 17.0,
                10.0 / 19.0,
                10.0 / 20.0,
                11.0 / 22.0,
            ],
            dtype=torch.double,
        )
        exp_pos_recall = torch.tensor(
            [
                3.0 / 11.0,
                4.0 / 11.0,
                6.0 / 11.0,
                6.0 / 11.0,
                8.0 / 11.0,
                9.0 / 11.0,
                10.0 / 11.0,
                10.0 / 11.0,
                10.0 / 11.0,
                11.0 / 11.0,
            ],
            dtype=torch.double,
        )

        exp_neg_prec = torch.tensor(
            [
                1.0 / 2.0,
                2.0 / 3.0,
                4.0 / 5.0,
                5.0 / 7.0,
                6.0 / 9.0,
                6.0 / 11.0,
                8.0 / 13.0,
                8.0 / 15.0,
                9.0 / 17.0,
                10.0 / 21.0,
                11.0 / 22.0,
            ],
            dtype=torch.double,
        )
        exp_neg_recall = torch.tensor(
            [
                1.0 / 11.0,
                2.0 / 11.0,
                4.0 / 11.0,
                5.0 / 11.0,
                6.0 / 11.0,
                6.0 / 11.0,
                8.0 / 11.0,
                8.0 / 11.0,
                9.0 / 11.0,
                10.0 / 11.0,
                11.0 / 11.0,
            ],
            dtype=torch.double,
        )

        torch.testing.assert_allclose(pr_curves["prec"][1], exp_pos_prec)
        torch.testing.assert_allclose(pr_curves["prec"][0], exp_neg_prec)

        torch.testing.assert_allclose(pr_curves["recall"][1], exp_pos_recall)
        torch.testing.assert_allclose(pr_curves["recall"][0], exp_neg_recall)

        torch.testing.assert_allclose(
            pr_curves["ap"][1], torch.tensor(0.589678058127256).double()
        )
        torch.testing.assert_allclose(
            pr_curves["ap"][0], torch.tensor(0.6073388287292031).double()
        )

    def test_compute_pr_curves_fail(self):
        class_hist = self._get_base_class_hist()
        total_hist = self._get_base_total_hist()

        # invalid histograms
        with self.assertRaises(AssertionError):
            class_hist += torch.ones(class_hist.size(), dtype=torch.int64) * 100
            util.compute_pr_curves(class_hist, total_hist)

        # Doesn't accept numpy
        with self.assertRaises(AssertionError):
            util.compute_pr_curves(class_hist.numpy(), total_hist)

        with self.assertRaises(AssertionError):
            util.compute_pr_curves(class_hist, total_hist.numpy())

        # Longs only
        with self.assertRaises(AssertionError):
            util.compute_pr_curves(class_hist.float(), total_hist.float())

        # Bad tensor size
        with self.assertRaises(AssertionError):
            util.compute_pr_curves(class_hist.view(40, 1), total_hist)

        with self.assertRaises(AssertionError):
            util.compute_pr_curves(class_hist, total_hist.view(40, 1))

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

        invalid_gpu_copy_values = [1234, True, 1.0]
        for value in invalid_gpu_copy_values:
            with self.assertRaises(AttributeError):
                gpu_value = util.recursive_copy_to_gpu(value)

        invalid_gpu_copy_depth = [
            ((((tensor_a, tensor_b), tensor_b), tensor_b), tensor_b),
            {"tensor_map_a": {"tensor_map_b": {"tensor_map_c": {"tensor": tensor_a}}}},
            [[[[tensor_a, tensor_b], tensor_b], tensor_b], tensor_b],
            "abcd",  # Strings are sequences, includeing single char strings
        ]
        for value in invalid_gpu_copy_depth:
            with self.assertRaises(ValueError):
                gpu_value = util.recursive_copy_to_gpu(value, max_depth=3)

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
        trainer = LocalTrainer(use_gpu=False)
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
        use_gpu = torch.cuda.is_available()
        trainer = LocalTrainer(use_gpu=use_gpu)
        trainer.train(task)
        for reset_heads in [False, True]:
            task_2 = build_task(config)
            # prepare task_2 for the right device
            task_2.prepare(use_gpu=use_gpu)
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
