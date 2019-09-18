#!/usr/bin/env python3

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
from torch.utils.data._utils.signal_handling import (
    _remove_worker_pids,
    _set_SIGCHLD_handler,
    _set_worker_pids,
    _set_worker_signal_handlers,
)

from .signals import Signals


# constants:
MANAGER_STATUS_CHECK_INTERVAL = 0.5  # In seconds
NUM_SAMPLES_TO_PREFETCH = 3  # Chosen empirically


class MultiprocessIterationError(Exception):
    pass


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

    # run worker:
    while True:

        # try and get new job from queue:
        try:
            batch_idx = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue

        # get batch from dataset:
        if batch_idx == Signals.SHUTDOWN_WORKER:
            index_queue.close()
            data_queue.close()
            index_queue.join_thread()
            data_queue.join_thread()
            return

        if batch_idx == Signals.LAST_SAMPLE:
            break

        try:
            batch = dataset[batch_idx]

        # put batch or exception in data queue:
        except Exception:
            data_queue.put((batch_idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((batch_idx, batch))
            del batch

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


def process_errors(batch):
    """
    Helper function to remove errors from batch queue
    """
    if isinstance(batch, ExceptionWrapper):  # forward exceptions to main process
        raise batch.exc_type(batch.exc_msg)
    return batch


class _BaseAsyncDatasetIterator(object):
    """
    Basic async iterator. This iterator allows multiple workers to
    process images in parallel.  It returns the batches in the same
    order as the dataset and does not backfill corrupted /
    failure-to-fetch images.

    If mp_start_method is None, it defaults to current context. If
    set, then the dataloader will create new processes using the
    specified method: fork, spawn, forkserver.

    Note: forking can lead to difficult to debug issues if any code is
    using calling into C / using threads.

    Spawn / forkserver will generally be less error-prone but require that all
    datasets are serializable via pickle.
    """

    def __init__(
        self,
        dataset,
        worker_loop=_worker_loop,
        num_workers=0,
        pin_memory=False,
        seed=0,
        mp_start_method=None,
    ):

        # create base class:
        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # bookkeeping variables:
        self.send_idx = 0
        self.rcvd_idx = 0
        self.reorder_dict = {}
        self.batches_outstanding = 0
        self._index_queues_shutdown = [False] * num_workers
        self._data_queue_shutdown = False
        self._finished_sending = False

        # initialize multiprocessing queues:
        mp = multiprocessing
        if mp_start_method is not None:
            assert sys.version_info >= (3, 4), "Require python 3.4 or later"
            mp = multiprocessing.get_context(mp_start_method)
        self.index_queues = [mp.Queue() for _ in range(self.num_workers)]
        self.data_queue = mp.Queue(-1)

        # initialize workers:
        self.workers = []
        for i in range(self.num_workers):
            w = mp.Process(
                target=worker_loop,
                daemon=True,  # ensures that the worker exits on process exit
                args=(self.dataset, self.index_queues[i], self.data_queue, seed + i, i),
            )

            # start the worker:
            w.start()
            self.workers.append(w)  # prevents calling join() on unstarted process

        # update worker pids:
        _set_worker_pids(id(self), tuple(w.pid for w in self.workers))
        _set_SIGCHLD_handler()
        self.worker_pids_set = True

        self._prime_index_queue()

    def __next__(self):
        while True:
            # Get batches from data queue until self.rcvd_idx is
            # found or shutdown condition is reached
            if self.batches_outstanding == 0 and len(self.reorder_dict) == 0:
                break
            if self.rcvd_idx in self.reorder_dict:
                batch = self.reorder_dict.pop(self.rcvd_idx)
                batch = self._process_next_batch(batch)
                if self.pin_memory:
                    batch = pin_memory(batch)
                return batch
            if self.batches_outstanding > 0:
                idx, batch = self.data_queue.get()
                self.batches_outstanding -= 1
                self.reorder_dict[idx] = batch
                continue
            else:
                # If batches is 0 and self.rcvd_idx is not in reorder_dict
                # Then something is wrong with the iteration
                raise MultiprocessIterationError(
                    "Missing idx {} in reorder_dict".format(self.rcvd_idx)
                )

        self._shutdown_workers()
        raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        self._shutdown_workers()
        # If shutdown still hasn't happened, forcibly terminate processes
        for w in self.workers:
            if w.is_alive():
                logging.error(
                    "Worker still alive at iterator deletion time. Terminating."
                )
                w.terminate()
                w.join()
                logging.error("Worker terminated.")

    def _prime_index_queue(self):
        for _ in range(NUM_SAMPLES_TO_PREFETCH * self.num_workers):
            self._put_indices()

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        # For base iterator, batches outsanding should always be less
        # than prefetch queue size * number of workers
        assert self.batches_outstanding < NUM_SAMPLES_TO_PREFETCH * self.num_workers
        self._put_indices()
        return process_errors(batch)

    def _put_indices(self):
        # Once we reach the end of the dataset, put last sample signal
        # in for all workers
        if self.send_idx >= len(self.dataset):
            if not self._finished_sending:
                self._finished_sending = True
                for w in self.index_queues:
                    w.put(Signals.LAST_SAMPLE)
            return

        # put index in queue and update bookkeeping variables:
        worker = self.send_idx % self.num_workers
        self.index_queues[worker].put(self.send_idx)
        # Update send values
        self.send_idx += 1
        self.batches_outstanding += 1

    def _shutdown_workers(self):
        try:
            # empty all queues not shutdown yet. We do this queue by
            # queue in case an exception occurs during the shutdown
            # process during main loop. This is called a second time
            # at __del__
            for idx, q in enumerate(self.index_queues):
                if not self._index_queues_shutdown[idx]:
                    self._index_queues_shutdown[idx] = True
                    q.put(Signals.SHUTDOWN_WORKER)
                    # Wait up to interval for worker to shutdown
                    self.workers[idx].join(MANAGER_STATUS_CHECK_INTERVAL)

            if not self._data_queue_shutdown:
                try:
                    self._data_queue_shutdown = True
                    while not self.data_queue.empty():
                        self.data_queue.get()
                except (FileNotFoundError, ImportError):
                    pass

        # remove pids no matter what happens:
        finally:
            if hasattr(self, "worker_pids_set") and self.worker_pids_set:
                _remove_worker_pids(id(self))
                self.worker_pids_set = False
