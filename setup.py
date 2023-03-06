#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import sys

from setuptools import find_namespace_packages, find_packages, setup


if __name__ == "__main__":
    if sys.version_info < (3, 7):
        sys.exit("Sorry, Python >=3.7 is required for Classy Vision.")

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
        python_requires=">=3.7",
        packages=find_packages(exclude=("tests",))
        + find_namespace_packages(include=["hydra_plugins.*"]),
        install_requires=reqs.strip().split("\n"),
        extras_require={
            "dev": [
                "GitPython",
                "black>=23.1.0",
                "sphinx",
                "isort>=5.12.0",
                "bs4",
                "nbconvert>=7.2.9",
                "pre-commit",
                "parameterized",
                "fairscale>=0.4.13",
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
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
        ],
    )
