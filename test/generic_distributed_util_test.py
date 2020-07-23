#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from typing import Any, Dict, List

import torch
from classy_vision.generic.distributed_util import broadcast_object
from torch.multiprocessing import Event, Process, Queue


def init_and_run_process(
    rank, world_size, filename, fn, input, q, wait_event, backend="gloo"
):
    torch.distributed.init_process_group(
        backend, init_method=f"file://{filename}", rank=rank, world_size=world_size
    )
    r = fn(*input)
    q.put(r)

    wait_event.wait()
    return


def run_in_process_group(filename: str, calls: List[Dict[str, Any]]):
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    processes = []
    q = Queue()
    wait_event = Event()

    # run the remaining processes
    # for rank in range(world_size - 1):
    for rank, call in enumerate(calls):
        p = Process(
            target=init_and_run_process,
            args=(
                rank,
                call["world_size"],
                filename,
                call["function"],
                call["inputs"],
                q,
                wait_event,
            ),
        )
        p.start()
        processes.append(p)

    # fetch the results from the queue before joining, the background processes
    # need to be alive if the queue contains tensors. See
    # https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847/3  # noqa: B950
    results = []
    for _ in range(len(processes)):
        results.append(q.get())

    wait_event.set()

    for p in processes:
        p.join()
    return results


class TestDistributedUtil(unittest.TestCase):
    @staticmethod
    def _get_test_objects():
        return [
            {"a": 12, "b": [2, 3, 4], "tensor": torch.randn(10, 10)},
            None,
            {"tensor": torch.randn(10000, 10000)},  # 400 MB
        ]

    def test_broadcast_object(self):
        world_size = 3
        for obj in self._get_test_objects():
            filename = tempfile.NamedTemporaryFile(delete=True).name
            inputs = [None] * world_size
            inputs[0] = obj  # only the master worker has the object

            calls = [
                {"world_size": world_size, "function": broadcast_object, "inputs": [i]}
                for i in inputs
            ]
            results = run_in_process_group(filename, calls)

            # check that all replicas got identical objects
            self.assertEqual(len(results), world_size)

            for result in results:
                if isinstance(obj, dict):
                    for key in obj:
                        if key == "tensor":
                            self.assertTrue(torch.allclose(result[key], obj[key]))
                        else:
                            self.assertEqual(result[key], obj[key])
                else:
                    self.assertEqual(result, obj)

    def test_broadcast_object_pick_source(self):
        world_size = 3
        for obj in self._get_test_objects():
            filename = tempfile.NamedTemporaryFile(delete=True).name
            inputs = [None] * world_size
            source_rank = 1
            inputs[source_rank] = obj  # only the rank 1 worker has the object

            calls = [
                {
                    "world_size": world_size,
                    "function": broadcast_object,
                    "inputs": [i, source_rank],
                }
                for i in inputs
            ]
            results = run_in_process_group(filename, calls)

            # check that all replicas got identical objects
            self.assertEqual(len(results), world_size)
            for result in results:
                if isinstance(obj, dict):
                    for key in obj:
                        if key == "tensor":
                            self.assertTrue(torch.allclose(result[key], obj[key]))
                        else:
                            self.assertEqual(result[key], obj[key])
                else:
                    self.assertEqual(result, obj)
