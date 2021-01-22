#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest.mock as mock
from test.generic.config_utils import get_test_classy_task
from test.generic.hook_test_utils import HookTestBase

import torch
import tqdm
from classy_vision.hooks import ProgressBarHook
from classy_vision.tasks.classification_task import LastBatchInfo


class TestProgressBarHook(HookTestBase):
    def test_constructors(self) -> None:
        """
        Test that the hooks are constructed correctly.
        """
        config = {}
        self.constructor_test_helper(
            config=config, hook_type=ProgressBarHook, hook_registry_name="progress_bar"
        )

    @mock.patch("classy_vision.hooks.progress_bar_hook.tqdm")
    @mock.patch("classy_vision.hooks.progress_bar_hook.is_primary")
    def test_progress_bar(
        self, mock_is_primary: mock.MagicMock, mock_tqdm_pkg: mock.MagicMock
    ) -> None:
        """
        Tests that the progress bar is created, updated and destroyed correctly.
        """
        mock_progress_bar = mock.create_autospec(tqdm.tqdm, instance=True)
        mock_tqdm_pkg.tqdm.return_value = mock_progress_bar

        mock_is_primary.return_value = True

        task = get_test_classy_task()
        task.prepare()
        task.advance_phase()

        num_batches = task.num_batches_per_phase
        # make sure we are checking at least one batch
        self.assertGreater(num_batches, 0)

        # create a progress bar hook
        progress_bar_hook = ProgressBarHook()

        # progressbar.ProgressBar should be init-ed with num_batches
        progress_bar_hook.on_phase_start(task)
        phase_images = len(task.datasets[task.phase_type])
        mock_tqdm_pkg.tqdm.assert_called_once_with(
            total=phase_images, desc=task.phase_type, unit="images", leave=False
        )
        mock_tqdm_pkg.tqdm.reset_mock()

        # on_step should update the progress bar correctly
        for i in range(num_batches):
            # Fake a batch
            batch_size = 32
            task.last_batch = LastBatchInfo(
                loss=torch.empty(batch_size),
                output=torch.empty(batch_size),
                target=torch.empty(batch_size),
                sample={},
                step_data={},
            )
            progress_bar_hook.on_step(task)
            mock_progress_bar.update.assert_called_once_with(batch_size)
            mock_progress_bar.update.reset_mock()

        # check that even if the progress bar isn't created, the code doesn't
        # crash
        progress_bar_hook = ProgressBarHook()
        try:
            progress_bar_hook.on_step(task)
            progress_bar_hook.on_phase_end(task)
        except Exception as e:
            self.fail(
                "Received Exception when on_phase_start() isn't called: {}".format(e)
            )
        mock_tqdm_pkg.ProgressBar.assert_not_called()

        # check that a progress bar is not created if is_primary() returns False
        mock_is_primary.return_value = False
        progress_bar_hook = ProgressBarHook()
        try:
            progress_bar_hook.on_phase_start(task)
            progress_bar_hook.on_step(task)
            progress_bar_hook.on_phase_end(task)
        except Exception as e:
            self.fail("Received Exception when is_primary() is False: {}".format(e))
        self.assertIsNone(progress_bar_hook.progress_bar)
        mock_tqdm_pkg.ProgressBar.assert_not_called()
