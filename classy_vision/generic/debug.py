#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys


def debug_info(type, value, tb):
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import pdb
        import traceback

        traceback.print_exception(type, value, tb)
        print
        pdb.post_mortem(tb)
