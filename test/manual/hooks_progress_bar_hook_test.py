#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock as mock
from test.generic.config_utils import get_test_classy_task

import progressbar
from classy_vision.hooks import ProgressBarHook


class TestProgressBarHook(unittest.TestCase):
    @mock.patch("classy_vision.hooks.progress_bar_hook.progressbar")
    @mock.patch("classy_vision.hooks.progress_bar_hook.is_master")
    def test_progress_bar(
        self, mock_is_master: mock.MagicMock, mock_progressbar_pkg: mock.MagicMock
    ) -> None:
        """
        Tests that the progress bar is created, updated and destroyed correctly.
        """
        mock_progress_bar = mock.create_autospec(progressbar.ProgressBar, instance=True)
        mock_progressbar_pkg.ProgressBar.return_value = mock_progress_bar

        mock_is_master.return_value = True

        local_variables = {}

        task = get_test_classy_task()
        task.prepare()
        task.advance_phase()

        num_batches = task.num_batches_per_phase
        # make sure we are checking at least one batch
        self.assertGreater(num_batches, 0)

        # create a progress bar hook
        progress_bar_hook = ProgressBarHook()

        # progressbar.ProgressBar should be init-ed with num_batches
        progress_bar_hook.on_phase_start(task, local_variables)
        mock_progressbar_pkg.ProgressBar.assert_called_once_with(num_batches)
        mock_progress_bar.start.assert_called_once_with()
        mock_progress_bar.start.reset_mock()
        mock_progressbar_pkg.ProgressBar.reset_mock()

        # on_step should update the progress bar correctly
        for i in range(num_batches):
            progress_bar_hook.on_step(task, local_variables)
            mock_progress_bar.update.assert_called_once_with(i + 1)
            mock_progress_bar.update.reset_mock()

        # check that even if on_step is called again, the progress bar is
        # only updated with num_batches
        for _ in range(num_batches):
            progress_bar_hook.on_step(task, local_variables)
            mock_progress_bar.update.assert_called_once_with(num_batches)
            mock_progress_bar.update.reset_mock()

        # finish should be called on the progress bar
        progress_bar_hook.on_phase_end(task, local_variables)
        mock_progress_bar.finish.assert_called_once_with()
        mock_progress_bar.finish.reset_mock()

        # check that even if the progress bar isn't created, the code doesn't
        # crash
        progress_bar_hook = ProgressBarHook()
        try:
            progress_bar_hook.on_step(task, local_variables)
            progress_bar_hook.on_phase_end(task, local_variables)
        except Exception as e:
            self.fail(
                "Received Exception when on_phase_start() isn't called: {}".format(e)
            )
        mock_progressbar_pkg.ProgressBar.assert_not_called()

        # check that a progress bar is not created if is_master() returns False
        mock_is_master.return_value = False
        progress_bar_hook = ProgressBarHook()
        try:
            progress_bar_hook.on_phase_start(task, local_variables)
            progress_bar_hook.on_step(task, local_variables)
            progress_bar_hook.on_phase_end(task, local_variables)
        except Exception as e:
            self.fail("Received Exception when is_master() is False: {}".format(e))
        self.assertIsNone(progress_bar_hook.progress_bar)
        mock_progressbar_pkg.ProgressBar.assert_not_called()
