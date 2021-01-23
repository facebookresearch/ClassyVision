# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from classy_vision.configuration import ClassyConfigDict


class ClassyConfigDictTest(unittest.TestCase):
    def test_dict(self):
        d = ClassyConfigDict(a=1, b=[1, 2, "3"])
        d["c"] = [4]
        d["d"] = {"a": 2}
        self.assertEqual(d, {"a": 1, "b": [1, 2, "3"], "c": [4], "d": {"a": 2}})
        self.assertIsInstance(d, ClassyConfigDict)
        self.assertIsInstance(d["d"], ClassyConfigDict)

    def test_freezing(self):
        d = ClassyConfigDict(a=1, b=2)
        d.freeze()
        # resetting an already existing key
        with self.assertRaises(TypeError):
            d["a"] = 3
        # adding a new key
        with self.assertRaises(TypeError):
            d["f"] = 3

    def test_unused_keys(self):
        d = ClassyConfigDict(
            a=1,
            b=[
                1,
                2,
                {
                    "c": {"a": 2},
                    "d": 4,
                    "e": {"a": 1, "b": 2},
                    "f": {"a": 1, "b": {"c": 2}},
                },
            ],
        )

        all_keys = {
            "a",
            "b",
            "b.2.c",
            "b.2.c.a",
            "b.2.d",
            "b.2.e",
            "b.2.f",
            "b.2.e.a",
            "b.2.e.b",
            "b.2.f.a",
            "b.2.f.b",
            "b.2.f.b.c",
        }

        def test_func(**kwargs):
            return None

        for _ in range(2):
            expected_unused_keys = all_keys.copy()
            self.assertSetEqual(set(d.unused_keys()), expected_unused_keys)

            _ = d["a"]
            expected_unused_keys.remove("a")
            self.assertSetEqual(set(d.unused_keys()), expected_unused_keys)

            _ = d["b"][2].get("d")
            expected_unused_keys.remove("b")
            expected_unused_keys.remove("b.2.d")
            self.assertSetEqual(set(d.unused_keys()), expected_unused_keys)

            _ = d["b"][2]["e"]
            expected_unused_keys.remove("b.2.e")
            self.assertSetEqual(set(d.unused_keys()), expected_unused_keys)

            _ = d["b"][2]["e"].items()
            expected_unused_keys.remove("b.2.e.a")
            expected_unused_keys.remove("b.2.e.b")
            self.assertSetEqual(set(d.unused_keys()), expected_unused_keys)

            _ = d["b"][2]["f"]
            expected_unused_keys.remove("b.2.f")
            self.assertSetEqual(set(d.unused_keys()), expected_unused_keys)

            test_func(**d["b"][2]["f"])
            expected_unused_keys.remove("b.2.f.a")
            expected_unused_keys.remove("b.2.f.b")
            self.assertSetEqual(set(d.unused_keys()), expected_unused_keys)

            _ = copy.deepcopy(d)
            expected_unused_keys.remove("b.2.c")
            expected_unused_keys.remove("b.2.c.a")
            expected_unused_keys.remove("b.2.f.b.c")
            self.assertSetEqual(set(d.unused_keys()), expected_unused_keys)

            d.reset_tracking()

    def test_to_dict(self):
        d = {
            "a": 1,
            "b": [
                1,
                2,
                {
                    "c": {"a": 2},
                    "d": 4,
                    "e": {"a": 1, "b": 2},
                    "f": {"a": 1, "b": {"c": 2}},
                },
            ],
        }
        classy_config_dict = ClassyConfigDict(**d)
        self.assertEqual(d, classy_config_dict.to_dict())
