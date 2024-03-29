#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import traceback
from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_task import ClassyTask


FILE_ROOT = Path(__file__).parent


TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()
TASK_REGISTRY_TB = {}
TASK_CLASS_NAMES_TB = {}


def build_task(config):
    """Builds a ClassyTask from a config.

    This assumes a 'name' key in the config which is used to determine what
    task class to instantiate. For instance, a config `{"name": "my_task",
    "foo": "bar"}` will find a class that was registered as "my_task"
    (see :func:`register_task`) and call .from_config on it."""

    task = TASK_REGISTRY[config["name"]].from_config(config)

    return task


def register_task(name):
    """Registers a ClassyTask subclass.

    This decorator allows Classy Vision to instantiate a subclass of ClassyTask
    from a configuration file, even if the class itself is not part of the
    Classy Vision framework. To use it, apply this decorator to a ClassyTask
    subclass, like this:

    .. code-block:: python

        @register_task('my_task')
        class MyTask(ClassyTask):
            ...

    To instantiate a task from a configuration file, see :func:`build_task`."""

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            msg = "Cannot register duplicate task ({}). Already registered at \n{}\n"
            raise ValueError(msg.format(name, TASK_REGISTRY_TB[name]))
        if not issubclass(cls, ClassyTask):
            raise ValueError(
                "Task ({}: {}) must extend ClassyTask".format(name, cls.__name__)
            )
        if cls.__name__ in TASK_CLASS_NAMES:
            msg = (
                "Cannot register task with duplicate class name({})."
                + "Previously registered at \n{}\n"
            )
            raise ValueError(
                msg.format(cls.__name__, TASK_CLASS_NAMES_TB[cls.__name__])
            )
        tb = "".join(traceback.format_stack())
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        TASK_REGISTRY_TB[name] = tb
        TASK_CLASS_NAMES_TB[cls.__name__] = tb
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
