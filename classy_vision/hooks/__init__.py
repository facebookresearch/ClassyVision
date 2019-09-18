#!/usr/bin/env python3

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent


# automatically import any Python files in the hooks/ directory
import_all_modules(FILE_ROOT, "classy_vision.hooks")
