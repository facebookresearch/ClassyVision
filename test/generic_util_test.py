#!/usr/bin/env python3

import unittest
import unittest.mock as mock
from pathlib import Path

import classy_vision.generic.util as util
import torch


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
        print(class_hist)
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
            [6.0 / 9.0, 8.0 / 13.0, 9.0 / 15.0, 10.0 / 17.0, 11.0 / 22.0]
        ).double()
        exp_pos_recall = torch.tensor(
            [6.0 / 11.0, 8.0 / 11.0, 9.0 / 11.0, 10.0 / 11.0, 11.0 / 11.0]
        ).double()

        exp_neg_prec = torch.tensor(
            [4.0 / 5.0, 5.0 / 7.0, 6.0 / 9.0, 8.0 / 13.0, 9.0 / 17.0, 11.0 / 22.0]
        ).double()
        exp_neg_recall = torch.tensor(
            [4.0 / 11.0, 5.0 / 11.0, 6.0 / 11.0, 8.0 / 11.0, 9.0 / 11.0, 11.0 / 11.0]
        ).double()

        torch.testing.assert_allclose(pr_curves["prec"][1], exp_pos_prec)
        torch.testing.assert_allclose(pr_curves["prec"][0], exp_neg_prec)

        torch.testing.assert_allclose(pr_curves["recall"][1], exp_pos_recall)
        torch.testing.assert_allclose(pr_curves["recall"][0], exp_neg_recall)

        torch.testing.assert_allclose(
            pr_curves["ap"][1], torch.tensor(0.6389071941375732).double()
        )
        torch.testing.assert_allclose(
            pr_curves["ap"][0], torch.tensor(0.68468004465103150).double()
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
            "num_phases": 12,
            "criterion": {"name": "test_loss"},
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
