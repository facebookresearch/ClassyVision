# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
from collections.abc import MutableMapping, Mapping

from .config_error import ConfigUnusedKeysError


class ClassyConfigDict(MutableMapping):
    """Mapping which can be made immutable. Also supports tracking unused keys."""

    def __init__(self, *args, **kwargs):
        """Create a ClassyConfigDict.

        Supports the same API as a dict and recursively converts all dicts to
        ClassyConfigDicts.
        """

        # NOTE: Another way to implement this would be to subclass dict, but since dict
        # is a built-in, it isn't treated like a regular MutableMapping, and calls like
        # func(**map) are handled mysteriously, probably interpreter dependent.
        # The downside with this implementation is that this isn't a full dict and is
        # just a mapping, which means some features like JSON serialization don't work

        self._dict = dict(*args, **kwargs)
        self._frozen = False
        self._keys_read = set()
        for k, v in self._dict.items():
            self._dict[k] = self._from_dict(v)

    @classmethod
    def _from_dict(cls, obj):
        """Recursively convert all dicts inside obj to ClassyConfigDicts"""

        if isinstance(obj, Mapping):
            obj = ClassyConfigDict({k: cls._from_dict(v) for k, v in obj.items()})
        elif isinstance(obj, (list, tuple)):
            # tuples are also converted to lists
            obj = [cls._from_dict(v) for v in obj]
        return obj

    def to_dict(self):
        """Return a vanilla Python dict, converting dicts recursively"""
        return self._to_dict(self)

    @classmethod
    def _to_dict(cls, obj):
        """Recursively convert obj to vanilla Python dicts"""
        if isinstance(obj, ClassyConfigDict):
            obj = {k: cls._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # tuples are also converted to lists
            obj = [cls._to_dict(v) for v in obj]
        return obj

    def keys(self):
        return self._dict.keys()

    def items(self):
        self._keys_read.update(self._dict.keys())
        return self._dict.items()

    def values(self):
        self._keys_read.update(self._dict.keys())
        return self._dict.values()

    def pop(self, key, default=None):
        return self._dict.pop(key, default)

    def popitem(self):
        return self._dict.popitem()

    def clear(self):
        self._dict.clear()

    def update(self, *args, **kwargs):
        if self._frozen:
            raise TypeError("Frozen ClassyConfigDicts do not support updates")
        self._dict.update(*args, **kwargs)

    def setdefault(self, key, default=None):
        return self._dict.setdefault(key, default)

    def __contains__(self, key):
        return key in self._dict

    def __eq__(self, obj):
        return self._dict == obj

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, key):
        self._keys_read.add(key)
        return self._dict.__getitem__(key)

    def __iter__(self):
        return iter(self._dict)

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)

    def __repr__(self):
        return repr(self._dict)

    def get(self, key, default=None):
        if key in self._dict.keys():
            self._keys_read.add(key)
        return self._dict.get(key, default)

    def __copy__(self):
        ret = ClassyConfigDict()
        for key, value in self._dict.items():
            self._keys_read.add(key)
            ret._dict[key] = value

    def copy(self):
        return self.__copy__()

    def __deepcopy__(self, memo=None):
        # for deepcopies we mark all the keys and sub-keys as read
        ret = ClassyConfigDict()
        for key, value in self._dict.items():
            self._keys_read.add(key)
            ret._dict[key] = copy.deepcopy(value)
        return ret

    def __setitem__(self, key, value):
        if self._frozen:
            raise TypeError("Frozen ClassyConfigDicts do not support assignment")
        if isinstance(value, dict) and not isinstance(value, ClassyConfigDict):
            value = ClassyConfigDict(value)
        self._dict.__setitem__(key, value)

    def __delitem__(self, key):
        if self._frozen:
            raise TypeError("Frozen ClassyConfigDicts do not support key deletion")
        del self._dict[key]

    def _freeze(self, obj):
        if isinstance(obj, Mapping):
            assert isinstance(obj, ClassyConfigDict), f"{obj} is not a ClassyConfigDict"
            obj._frozen = True
            for value in obj.values():
                self._freeze(value)
        elif isinstance(obj, list):
            for value in obj:
                self._freeze(value)

    def _reset_tracking(self, obj):
        if isinstance(obj, Mapping):
            assert isinstance(obj, ClassyConfigDict), f"{obj} is not a ClassyConfigDict"
            obj._keys_read = set()
            for value in obj._dict.values():
                self._reset_tracking(value)
        elif isinstance(obj, list):
            for value in obj:
                self._reset_tracking(value)

    def _unused_keys(self, obj):
        unused_keys = []
        if isinstance(obj, Mapping):
            assert isinstance(obj, ClassyConfigDict), f"{obj} is not a ClassyConfigDict"
            unused_keys = [key for key in obj._dict.keys() if key not in obj._keys_read]
            for key, value in obj._dict.items():
                unused_keys += [
                    f"{key}.{subkey}" for subkey in self._unused_keys(value)
                ]
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                unused_keys += [f"{i}.{subkey}" for subkey in self._unused_keys(value)]
        return unused_keys

    def freeze(self):
        """Freeze the ClassyConfigDict to disallow mutations"""
        self._freeze(self)

    def reset_tracking(self):
        """Reset key tracking"""
        self._reset_tracking(self)

    def unused_keys(self):
        """Fetch all the unused keys"""
        return self._unused_keys(self)

    def check_unused_keys(self):
        """Raise if the config has unused keys"""
        unused_keys = self.unused_keys()
        if unused_keys:
            raise ConfigUnusedKeysError(unused_keys)
