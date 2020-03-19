#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock as mock
from test.generic.config_utils import get_test_classy_task, get_test_model_configs

from classy_vision.hooks import ModelTensorboardHook, build_hook
from classy_vision.models import build_model
from tensorboardX import SummaryWriter


class TestModelTensorboardHook(unittest.TestCase):
    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        config = {"name": "model_tensorboard", "tensorboard_summary_writer": {}}

        hook1 = ModelTensorboardHook(tb_writer=SummaryWriter())
        hook2 = ModelTensorboardHook.from_config(config)
        hook3 = build_hook(config)

        self.assertTrue(isinstance(hook1, ModelTensorboardHook))
        self.assertTrue(isinstance(hook2, ModelTensorboardHook))
        self.assertTrue(isinstance(hook3, ModelTensorboardHook))

    @mock.patch("classy_vision.hooks.model_tensorboard_hook.is_master")
    def test_writer(self, mock_is_master_func: mock.MagicMock) -> None:
        """
        Tests that the tensorboard writer calls SummaryWriter with the model
        iff is_master() is True.
        """
        mock_summary_writer = mock.create_autospec(SummaryWriter, instance=True)

        task = get_test_classy_task()
        task.prepare()

        for master in [False, True]:
            mock_is_master_func.return_value = master
            model_configs = get_test_model_configs()

            for model_config in model_configs:
                model = build_model(model_config)
                task.base_model = model

                # create a model tensorboard hook
                model_tensorboard_hook = ModelTensorboardHook(mock_summary_writer)

                model_tensorboard_hook.on_start(task)

                if master:
                    # SummaryWriter should have been init-ed with the correct
                    # add_graph should be called once with model as the first arg
                    mock_summary_writer.add_graph.assert_called_once()
                    self.assertEqual(
                        mock_summary_writer.add_graph.call_args[0][0], model
                    )
                else:
                    # add_graph shouldn't be called since is_master() is False
                    mock_summary_writer.add_graph.assert_not_called()
                mock_summary_writer.reset_mock()
