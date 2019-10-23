#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_task import ClassyTask


FILE_ROOT = Path(__file__).parent


TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


def build_task(config, args, **kwargs):
    # allow some command-line options to override configuration:
    if "test_only" not in config:
        config["test_only"] = args.test_only
    return TASK_REGISTRY[config["name"]].from_config(config)


def register_task(name):
    """
    New tasks can be added to classy_vision with the
    :func:`~classy_vision.tasks.register_task` function decorator.
    For example::

        @register_task('classification')
        class ClassificationTask(ClassyTask):
            (...)
    .. note::

        All Tasks must implement the :class:`~classy_vision.tasks.ClassyTask`
        interface.

    Please see the

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, ClassyTask):
            raise ValueError(
                "Task ({}: {}) must extend ClassyTask".format(name, cls.__name__)
            )
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_task_cls


from .classification_task import ClassificationTask  # isort:skip
from .fine_tuning_task import FineTuningTask  # isort:skip

__all__ = [
    "ClassyTask",
    "FineTuningTask",
    "build_task",
    "register_task",
    "ClassificationTask",
]

# automatically import any Python files in the tasks/ directory
import_all_modules(FILE_ROOT, "classy_vision.tasks")
