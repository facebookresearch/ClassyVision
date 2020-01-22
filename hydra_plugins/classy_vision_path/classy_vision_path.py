#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from hydra.plugins import SearchPathPlugin


class ClassyVisionPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path):
        search_path.append("classy_vision", "pkg://classy_vision.hydra.conf")
