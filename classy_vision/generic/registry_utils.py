#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import os
import sys
from pathlib import Path


def import_all_modules(root: str, base_module: str) -> None:
    for file in os.listdir(root):
        if file.endswith((".py", ".pyc")) and not file.startswith("_"):
            module = file[: file.find(".py")]
            if module not in sys.modules:
                module_name = ".".join([base_module, module])
                importlib.import_module(module_name)


def import_all_packages_from_directory(root: str) -> None:
    """Automatically imports all packages under the root directory.

    For instance, if your directories look like:
        root / foo / __init__.py
        root / foo / abc.py
        root / bar.py
        root / baz / xyz.py

    This function will import the package foo, but not bar or baz."""

    for file in os.listdir(root):
        # Try to import each file in the directory. Our previous implementation
        # would look for directories here and see if there's a __init__.py
        # under that directory, but that turns out to be unreliable while
        # running on AWS: EFS filesystems cache metadata bits so the directory
        # and existence checks fail even when the import succeeds. We should
        # find a better workaround eventually, but this will do for now.
        try:
            file = Path(file)
            module_name = file.name
            # Dots have special meaning in Python packages -- it's a relative
            # import or a subpackage. Skip these.
            if "." not in module_name and module_name not in sys.modules:
                logging.debug(f"Automatically importing {module_name}")
                importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass
