#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys

from setuptools import find_packages, setup


if __name__ == "__main__":
    if sys.version_info < (3, 6):
        sys.exit("Sorry, Python >=3.6 is required for Classy Vision.")

    with open("README.md", encoding="utf8") as f:
        readme = f.read()

    with open("LICENSE") as f:
        license = f.read()

    with open("requirements.txt") as f:
        reqs = f.read()

    setup(
        name="ClassyVision",
        version="0.1.0",
        description="An end-to-end computer vision framework.",
        long_description=readme,
        url="TBD",
        license=license,
        python_requires=">=3.6",
        packages=find_packages(exclude=("configs", "tests")),
        install_requires=reqs.strip().split("\n"),
        include_package_data=True,
        test_suite="test.suites.unittests",
    )
