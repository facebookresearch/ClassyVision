#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import socket
import subprocess
import sys
import time
from argparse import REMAINDER, ArgumentParser
from contextlib import closing

import ray


class NodeLaunchActor:
    """Ray actor. The code here will run in each node allocated by Ray."""

    def run(self, master_addr, master_port, node_rank, dist_world_size, args):
        processes = []

        # set PyTorch distributed related environmental variables
        current_env = os.environ.copy()
        current_env["MASTER_ADDR"] = master_addr
        current_env["MASTER_PORT"] = str(master_port)
        current_env["WORLD_SIZE"] = str(dist_world_size)

        if "OMP_NUM_THREADS" not in os.environ and args.nproc_per_node > 1:
            current_env["OMP_NUM_THREADS"] = str(1)
            print(
                "*****************************************\n"
                "Setting OMP_NUM_THREADS environment variable for each process "
                "to be {} in default, to avoid your system being overloaded, "
                "please further tune the variable for optimal performance in "
                "your application as needed. \n"
                "*****************************************".format(
                    current_env["OMP_NUM_THREADS"]
                )
            )

        # Set the init_method and rank of the process for distributed training.
        for local_rank in range(0, args.nproc_per_node):
            # each process's rank
            dist_rank = args.nproc_per_node * node_rank + local_rank
            current_env["RANK"] = str(dist_rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            # spawn the processes
            with_python = not args.no_python
            cmd = []
            if with_python:
                cmd = [sys.executable, "-u"]
                if args.module:
                    cmd.append("-m")
            else:
                if not args.use_env:
                    raise ValueError(
                        "When using the '--no_python' flag, "
                        "you must also set the '--use_env' flag."
                    )
                if args.module:
                    raise ValueError(
                        "Don't use both the '--no_python' flag"
                        "and the '--module' flag at the same time."
                    )

            cmd.append(args.training_script)

            if not args.use_env:
                cmd.append("--local_rank={}".format(local_rank))

            cmd.extend(args.training_script_args)
            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)

        for process in processes:
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=process.returncode, cmd=cmd
                )

    def get_node_ip(self):
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]


def wait_for_gpus(world_size, timeout_secs=3600):
    n_gpus = int(ray.cluster_resources().get("GPU", 0))
    elapsed_time = 0
    while n_gpus < world_size:
        logging.warning(
            f"Not enough GPUs available ({n_gpus} available,"
            f"need {world_size}), waiting 10 seconds"
        )
        time.sleep(10)
        elapsed_time += 10
        if elapsed_time > timeout_secs:
            raise RuntimeError("Timeout: could not find enough GPUs")
        n_gpus = int(ray.cluster_resources().get("GPU", 0))


def parse_args():
    """Helper function parsing the command line options.
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="Classy Vision distributed training launch "
        "helper utility that will spawn up multiple nodes using Ray"
    )

    # Optional arguments for the launch helper
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="The number of nodes to use for distributed training",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="The number of processes to launch on each node, "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU.",
    )
    parser.add_argument(
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'."
        "If set to True, the script will not pass "
        "--local_rank as argument, and will instead set LOCAL_RANK.",
    )
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        "'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the training script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )
    # Ray-related arguments
    group = parser.add_argument_group("Ray related arguments")
    group.add_argument("--ray-address", default="auto", type=str)

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    ray.init(address=args.ray_address)

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    wait_for_gpus(dist_world_size)

    # Set up Ray distributed actors.
    actor = ray.remote(num_cpus=1, num_gpus=args.nproc_per_node)(NodeLaunchActor)
    workers = [actor.remote() for i in range(args.nnodes)]

    # Set worker 0 as the master
    master_addr = ray.get(workers[0].get_node_ip.remote())
    master_port = ray.get(workers[0].find_free_port.remote())

    unfinished = [
        worker.run.remote(
            master_addr=master_addr,
            master_port=master_port,
            node_rank=i,
            dist_world_size=dist_world_size,
            args=args,
        )
        for i, worker in enumerate(workers)
    ]

    try:
        while len(unfinished) > 0:
            finished, unfinished = ray.wait(unfinished)
            finished = ray.get(finished)
    except Exception as inst:
        logging.exception("An error occurred:")

    ray.shutdown()


if __name__ == "__main__":
    main()
