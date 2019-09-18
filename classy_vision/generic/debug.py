#!/usr/bin/env python3

import sys


def debug_info(type, value, tb):
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        import pdb

        traceback.print_exception(type, value, tb)
        print
        pdb.post_mortem(tb)
