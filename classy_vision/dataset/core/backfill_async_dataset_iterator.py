#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# dependencies:
import logging
import queue
import random
import signal
import sys

import torch
import torch.multiprocessing as multiprocessing
from classy_vision.generic.util import ExceptionWrapper
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils.signal_handling import _set_worker_signal_handlers

from .base_async_dataset_iterator import (
    NUM_SAMPLES_TO_PREFETCH,
    _BaseAsyncDatasetIterator,
    process_errors,
)
from .signals import Signals


# constants:
MANAGER_STATUS_CHECK_INTERVAL = 0.5


def recursive_batchsize_per_replica(batch):
    """
    Recursively searches batch to find tensors. Returns size of first
    dimension in tensor. Throws if multiple tensors are found and
    their size is different.
    """
    if batch is None:
        return 0

    if torch.is_tensor(batch):
        if batch.numel() == 0 or batch.dim() == 0:
            return 0
        return batch.size()[0]
    elif isinstance(batch, dict):
        prev_size = None
        for val in batch.values():
            size = recursive_batchsize_per_replica(val)
            assert prev_size is None or prev_size == size
            prev_size = size

        assert prev_size is not None, "Bottom level of batch must be tensor"
        return prev_size
    elif isinstance(batch, (list, tuple)):
        prev_size = None
        for val in batch:
            size = recursive_batchsize_per_replica(val)
            assert prev_size is None or prev_size == size
            prev_size = size

        assert prev_size is not None, "Bottom level of batch must be tensor"
        return prev_size

    raise TypeError(
        "Improper type passed to recursive batchsize_per_replica: {}".format(
            type(batch)
        )
    )


def backfill_batch(batch, next_batch, batchsize_per_replica):
    """
    Takes two (potentially unfinished) batches and returns [finished_batch,
    unfinished_batch].

    A finished batch has size, batchsize_per_replica, and an unfinished batch has
    size less than batchsize_per_replica.  The finished batch is produced by
    concatenating the two batches and unfinished_batch is whatever is
    left over.

    Currently this assumes that the batching happens across the
    first dimension in the tensor.

    TODO: we should consider making this more general by allowing the
    user to specify a dimension to concat across.
    """
    # If next batch is empty / None, return
    if next_batch is None or len(next_batch) == 0:
        return None, batch

    assert batch is None or isinstance(batch, type(next_batch)), (
        "batch and next_batch must be of same type, instead: (%s, %s)"
        % (type(batch), type(next_batch))
    )

    if torch.is_tensor(next_batch):
        batch_len = batch.size()[0] if batch is not None else 0
        next_batch_len = next_batch.size()[0]
        num_needed = batchsize_per_replica - batch_len
        assert (
            num_needed > 0 and next_batch_len <= batchsize_per_replica
        ), """
            Batches should always be smaller than batchsize_per_replica.
            Unfinished batches should be strictly smaller than batchsize_per_replica:
            Unfinished Batch len: %d, Next batch len: %d, batchsize_per_replica: %d
            """ % (
            batch_len,
            next_batch_len,
            batchsize_per_replica,
        )
        if next_batch.dim() == 0:
            next_batch.unsqueeze_(0)

        if batch is None:
            batch = torch.tensor([], dtype=next_batch.dtype)

        if num_needed <= next_batch_len:
            batch = torch.cat([batch, next_batch[:num_needed]])
            unfinished_batch = next_batch[num_needed:]
        else:
            unfinished_batch = torch.cat([batch, next_batch])
            batch = None

        if unfinished_batch.numel() == 0:
            unfinished_batch = None

    elif isinstance(next_batch, dict):
        unfinished_batch = {}
        assert (
            batch is None or batch.keys() == next_batch.keys()
        ), "batch must be empty or have same keys as next batch"
        if batch is None:
            batch = {key: None for key in next_batch.keys()}

        for key in next_batch.keys():
            batch[key], unfinished_batch[key] = backfill_batch(
                batch[key], next_batch[key], batchsize_per_replica
            )

    elif isinstance(next_batch, (list, tuple)):
        assert batch is None or len(batch) == len(
            next_batch
        ), "Batch and next_batch must have same length"
        unfinished_batch = [0] * len(next_batch)
        if batch is None:
            batch = [None] * len(next_batch)
        if isinstance(batch, tuple):
            batch = list(batch)

        for idx in range(len(next_batch)):
            batch[idx], unfinished_batch[idx] = backfill_batch(
                batch[idx], next_batch[idx], batchsize_per_replica
            )

        if isinstance(next_batch, tuple):
            batch = tuple(batch)
            unfinished_batch = tuple(unfinished_batch)

    else:
        # For all other types, raise error.
        raise TypeError(
            "Improper type {} passed to backfill batches".format(type(batch))
        )

    # Return batch / unfinished_batch, prefer to return None for batch
    # rather than nested Nones or empty bottom layer batches
    if recursive_batchsize_per_replica(batch) == 0:
        batch = None
    return batch, unfinished_batch


# function that implements what a worker ought to be doing:
def _worker_loop(dataset, index_queue, data_queue, seed, worker_id):

    # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
    # module's handlers are executed after Python returns from C low-level
    # handlers, likely when the same fatal signal happened again already.
    # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
    _set_worker_signal_handlers()

    # Ensure that child processes are killed if their parent dies;
    # otherwise the root process won't detect child process failure correctly.
    # This is an issue because the code uses "fork" instead of "spawn".
    # See T44295967 for details.
    multiprocessing._prctl_pr_set_pdeathsig(signal.SIGKILL)

    # make sure worker uses single thread and set seeds:
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    batchsize_per_replica = dataset.batchsize_per_replica

    # run worker:
    num_success = 0
    total_requested = 0
    unfinished_batch = None
    ids = []
    while True:

        # try and get new job from queue:
        try:
            batch_idx = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue

        # Check for signals
        if batch_idx == Signals.SHUTDOWN_WORKER:
            index_queue.close()
            data_queue.close()
            index_queue.join_thread()
            data_queue.join_thread()
            return

        if batch_idx == Signals.LAST_SAMPLE:
            break

        ids.append(batch_idx)
        # get batch from dataset:
        try:
            next_batch = dataset[batch_idx]

            # Logging for success rate
            num_success += recursive_batchsize_per_replica(next_batch)
            # Last batch may not be full, so this will be up to batchsize_per_replica off  # noqa
            total_requested += batchsize_per_replica

            batch, unfinished_batch = backfill_batch(
                unfinished_batch, next_batch, batchsize_per_replica
            )
        except Exception:
            # put batch or exception in data queue:
            data_queue.put((ExceptionWrapper(sys.exc_info()), ids))
            ids = []
        else:
            if batch is not None:
                data_queue.put((batch, ids))
                ids = []
                del batch

    # Pass last (potentially partial) batch to main process
    if recursive_batchsize_per_replica(unfinished_batch) > 0:
        data_queue.put((unfinished_batch, ids))
        ids = []
        del unfinished_batch

    # Worker has finished. Alert main process, print success stats
    # Print to warning because PIL is noisy on debug channel so we
    # have to suppress it, we reserve info for outputs, and this is
    # not an error. If we migrate off of PIL, debug would be a better
    # choice.
    success_rate = 1.0 * num_success / total_requested if total_requested > 0 else 0.0
    msg = """
    Fetch stats for worker %s:
    Successfully fetched %d samples out of approximately %d requested.
    Success rate for iteration: %f
    """ % (
        str(worker_id),
        num_success,
        total_requested,
        success_rate,
    )
    logging.warning(msg)
    data_queue.put((Signals.WORKER_DONE, []))

    # Wait for shutdown command from main process
    while True:
        try:
            sig = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue

        if sig == Signals.SHUTDOWN_WORKER:
            break

    index_queue.close()
    data_queue.close()
    index_queue.join_thread()
    data_queue.join_thread()


class _BackfillAsyncDatasetIterator(_BaseAsyncDatasetIterator):
    """
    This async iterator will backfill samples when a sample is missing
    because of corruption / failure-to-fetch or other errors.  Because
    backfilling samples means that we no longer have guarantees about
    the number of batches produced or which samples are in which
    batch, the ordering assumption is already partially broken.  In
    addition, the approach for synchronizing the ordering in the basic
    async iterator will not work since the batch ids are no longer
    deterministic.

    As such, this iterator is unordered between the different workers,
    but each worker still moves through its batches in the order
    provided by the dataset.
    """

    def __init__(
        self, dataset, num_workers=0, pin_memory=False, seed=0, mp_start_method=None
    ):
        super(_BackfillAsyncDatasetIterator, self).__init__(
            dataset, _worker_loop, num_workers, pin_memory, seed, mp_start_method
        )
        assert (
            hasattr(dataset, "batchsize_per_replica")
            and dataset.batchsize_per_replica > 0
        ), """The backfill iterators can only be used
            with datasets with a valid batchsize_per_replica"""
        self._num_finished_workers = 0
        self._unfinished_batches = []
        self._batchsize_per_replica = dataset.batchsize_per_replica
        self._skip_last = False
        try:
            self._skip_last = dataset.skip_last
        except AttributeError:
            logging.info(
                "Dataset has no skip_last object. "
                "Iteration will process all batches by default"
            )

    def __next__(self):

        # check if the next batch has already been generated; if so, return:
        num_tries = 0
        while True:
            if self._num_finished_workers == self.num_workers:
                self._shutdown_workers()
                break

            # Get batch from data queue
            try:
                batch, rcvd_ids = self.data_queue.get(
                    timeout=MANAGER_STATUS_CHECK_INTERVAL
                )
                self.batches_outstanding -= len(rcvd_ids)
                batch = self._process_next_batch(batch, len(rcvd_ids))
                if batch == Signals.WORKER_DONE:
                    self._num_finished_workers += 1
                    continue

                if (
                    recursive_batchsize_per_replica(batch)
                    == self._batchsize_per_replica
                ):
                    if self.pin_memory:
                        batch = pin_memory(batch)
                    return batch

                self._unfinished_batches.append(batch)

            except queue.Empty:
                num_tries += 1

                # If stalled for 10 tries add more data to each
                # worker's processing queue. Note,
                # MANAGER_STATUS_CHECK_INTERVAL is in seconds, so this
                # is 10 * MANAGER_STATUS_CHECK_INTERVAL secs of
                # waiting
                if num_tries % 10 == 0:
                    logging.error(
                        "After %d tries, data queue still empty...adding more data"
                        % num_tries
                    )
                    for _ in range(self.num_workers):
                        self._put_indices()

        while len(self._unfinished_batches) > 1:
            batch = self._unfinished_batches[-2]
            next_batch = self._unfinished_batches[-1]
            batch, unfinished_batch = backfill_batch(
                batch, next_batch, self._batchsize_per_replica
            )
            self._unfinished_batches[-2] = unfinished_batch
            self._unfinished_batches.pop()
            if recursive_batchsize_per_replica(batch) > 0:
                if self.pin_memory:
                    batch = pin_memory(batch)
                return batch

        if (
            not self._skip_last
            and len(self._unfinished_batches) > 0
            and recursive_batchsize_per_replica(self._unfinished_batches[0]) > 0
        ):
            batch = self._unfinished_batches.pop()
            if self.pin_memory:
                batch = pin_memory(batch)
            return batch

        raise StopIteration

    next = __next__  # Python 2 compatibility

    def _process_next_batch(self, batch, num_rcvd=1):
        for _ in range(num_rcvd):
            self.rcvd_idx += 1
            # Sometimes, if there are too many errors, we have to add
            # additional samples to prefetch queue...but we don't want
            # the size of the queue to grow outside of that situation
            if self.batches_outstanding < NUM_SAMPLES_TO_PREFETCH * self.num_workers:
                self._put_indices()
        return process_errors(batch)
