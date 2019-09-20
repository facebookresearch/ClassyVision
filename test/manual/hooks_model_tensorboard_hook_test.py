#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock as mock
from test.generic.config_utils import get_test_classy_task, get_test_model_configs

from classy_vision.hooks import ModelTensorboardHook
from classy_vision.models import build_model
from tensorboardX import SummaryWriter


class TestModelTensorboardHook(unittest.TestCase):
    @mock.patch("classy_vision.hooks.model_tensorboard_hook.is_master")
    @mock.patch("classy_vision.generic.visualize.SummaryWriter", autospec=True)
    def test_writer(
        self,
        mock_summary_writer_cls: mock.MagicMock,
        mock_is_master_func: mock.MagicMock,
    ) -> None:
        """
        Tests that the tensorboard writer calls SummaryWriter with the model
        iff is_master() is True.
        """
        # TODO (mannatsingh): get rid of pyfakefs
        # needed to mock out SummaryWriter since it doesn't work with pyfakefs
        mock_summary_writer = mock.create_autospec(SummaryWriter, instance=True)
        mock_summary_writer_cls.return_value = mock_summary_writer

        task = get_test_classy_task()
        state = task.build_initial_state()

        for master in [False, True]:
            mock_is_master_func.return_value = master
            model_configs = get_test_model_configs()
            local_variables = {}

            for model_config in model_configs:
                model = build_model(model_config)
                state.base_model = model

                # create a model tensorboard hook
                model_tensorboard_hook = ModelTensorboardHook()

                with self.assertLogs():
                    model_tensorboard_hook.on_start(state, local_variables)

                if master:
                    # SummaryWriter should have been init-ed with the correct
                    # log_dir kwarg
                    mock_summary_writer_cls.assert_called_once()
                    self.assertEqual(
                        mock_summary_writer_cls.call_args[1]["log_dir"],
                        model_tensorboard_hook.tensorboard_dir,
                    )
                    # add_graph should be called once with model as the first arg
                    mock_summary_writer.add_graph.assert_called_once()
                    self.assertEqual(
                        mock_summary_writer.add_graph.call_args[0][0], model
                    )
                else:
                    # add_graph shouldn't be called since is_master() is False
                    mock_summary_writer.add_graph.assert_not_called()
                mock_summary_writer_cls.reset_mock()
                mock_summary_writer.reset_mock()
