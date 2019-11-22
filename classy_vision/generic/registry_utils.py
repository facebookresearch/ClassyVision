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


def import_all_modules(root, base_module):
    for file in os.listdir(root):
        if file.endswith((".py", ".pyc")) and not file.startswith("_"):
            module = file[: file.find(".py")]
            if module not in sys.modules:
                module_name = ".".join([base_module, module])
                importlib.import_module(module_name)


def import_all_packages_from_directory(root):
    """Automatically imports all packages under the root directory.

    For instance, if your directories look like:
        root / foo / __init__.py
        root / foo / abc.py
        root / bar.py
        root / baz / xyz.py

    This function will import the package foo, but not bar or baz."""

    for file in os.listdir(root):
        file = Path(file)
        if file.is_dir() and (file / "__init__.py").exists():
            module_name = file.name
            if module_name not in sys.modules:
                logging.debug(f"Automatically importing {module_name}")
                importlib.import_module(module_name)
