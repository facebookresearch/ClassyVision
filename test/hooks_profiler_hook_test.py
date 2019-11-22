#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock as mock
from test.generic.config_utils import get_test_classy_task, get_test_classy_video_task

from classy_vision.hooks import ProfilerHook


class TestProfilerHook(unittest.TestCase):
    @mock.patch("torch.autograd.profiler.profile", auto_spec=True)
    @mock.patch("classy_vision.hooks.profiler_hook.summarize_profiler_info")
    def test_profiler(
        self,
        mock_summarize_profiler_info: mock.MagicMock,
        mock_profile_cls: mock.MagicMock,
    ) -> None:
        """
        Tests that a profile instance is returned by the profiler
        and that the profiler actually ran.
        """
        mock_summarize_profiler_info.return_value = ""

        mock_profile = mock.MagicMock()
        mock_profile_returned = mock.MagicMock()
        mock_profile.__enter__.return_value = mock_profile_returned
        mock_profile_cls.return_value = mock_profile

        for task in [get_test_classy_task(), get_test_classy_video_task()]:
            task.prepare()
            local_variables = {}

            # create a model tensorboard hook
            profiler_hook = ProfilerHook()

            with self.assertLogs():
                profiler_hook.on_start(task, local_variables)

            # a new profile should be created with use_cuda=True
            mock_profile_cls.assert_called_once_with(use_cuda=True)
            mock_profile_cls.reset_mock()

            # summarize_profiler_info should have been called once with the profile
            mock_summarize_profiler_info.assert_called_once()
            profile = mock_summarize_profiler_info.call_args[0][0]
            mock_summarize_profiler_info.reset_mock()
            self.assertEqual(profile, mock_profile_returned)
