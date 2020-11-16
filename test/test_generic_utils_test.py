# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.generic.utils import ClassyTestCase

import torch


class TestClassyTestCase(unittest.TestCase):
    def test_assert_torch_all_close(self):
        test_fixture = ClassyTestCase()

        data = [1.1, 2.2]
        tensor_1 = torch.Tensor(data)

        # shouldn't raise an exception
        tensor_2 = tensor_1
        test_fixture.assertTorchAllClose(tensor_1, tensor_2)

        # should fail because tensors are not close
        tensor_2 = tensor_1 / 2
        with self.assertRaises(AssertionError):
            test_fixture.assertTorchAllClose(tensor_1, tensor_2)

        # should fail because tensor_2 is not a tensor
        tensor_2 = data
        with self.assertRaises(AssertionError):
            test_fixture.assertTorchAllClose(tensor_1, tensor_2)

        # should fail because tensor_1 is not a tensor
        tensor_1 = data
        tensor_2 = torch.Tensor(data)
        with self.assertRaises(AssertionError):
            test_fixture.assertTorchAllClose(tensor_1, tensor_2)
