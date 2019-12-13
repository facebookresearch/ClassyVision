#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import sys

from setuptools import find_packages, setup


if __name__ == "__main__":
    if sys.version_info < (3, 6):
        sys.exit("Sorry, Python >=3.6 is required for Classy Vision.")

    # get version string from module
    with open(
        os.path.join(os.path.dirname(__file__), "classy_vision/__init__.py"), "r"
    ) as f:
        version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(
            1
        )
        print("-- Building version " + version)

    with open("README.md", encoding="utf8") as f:
        readme = f.read()

    with open("requirements.txt") as f:
        reqs = f.read()

    setup(
        name="classy_vision",
        version=version,
        description="An end-to-end PyTorch framework for image and video classification.",
        long_description_content_type="text/markdown",
        long_description=readme,
        url="https://classyvision.ai",
        project_urls={
            "Documentation": "https://classyvision.ai",
            "Source": "https://github.com/facebookresearch/ClassyVision",
        },
        license="MIT License",
        python_requires=">=3.6",
        packages=find_packages(exclude=("tests",)),
        install_requires=reqs.strip().split("\n"),
        extras_require={
            "dev": [
                "GitPython",
                "black==19.3b0",
                "sphinx",
                "isort",
                "bs4",
                "nbconvert",
                "pre-commit",
            ]
        },
        package_data={"classy_vision": ["configs/*.json", "templates"]},
        data_files=[("classy_vision", ["classy_train.py"])],
        include_package_data=True,
        test_suite="test.suites.unittests",
        scripts=["bin/classy-project"],
        keywords=["deep learning", "pytorch", "AI"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
        ],
    )
