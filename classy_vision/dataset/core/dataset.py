#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Dataset(object):
    """
    Base dataset class.

    NOTE: Attribute accesses are passed on to self.dataset (if it exists), if
          not found in self.
    """

    # TODO: Should `dataset.func(*args, **kwargs)` change itself?

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset is an abstract class.")

    def __len__(self):
        raise NotImplementedError("Dataset is an abstract class.")

    def batch(self, *args, **kwargs):
        from .batch_dataset import BatchDataset

        return BatchDataset(self, *args, **kwargs)

    def resample(self, *args, **kwargs):
        from .resample_dataset import ResampleDataset

        return ResampleDataset(self, *args, **kwargs)

    def shuffle(self, *args, **kwargs):
        from .shuffle_dataset import ShuffleDataset

        return ShuffleDataset(self, *args, **kwargs)

    def transform(self, *args, **kwargs):
        from .transform_dataset import TransformDataset

        return TransformDataset(self, *args, **kwargs)

    def shard(self, *args, **kwargs):
        from .shard_dataset import ShardDataset

        return ShardDataset(self, *args, **kwargs)

    def iterator(self, *args, **kwargs):
        from .async_dataset_iterator import AsyncDatasetIterator
        from .dataset_iterator import DatasetIterator

        # Deleting args not used by AsyncDatasetIterator or DatasetIterator
        key = "current_phase_id"
        if key in kwargs:
            del kwargs[key]

        if "backfill_batches" not in kwargs:
            kwargs["backfill_batches"] = False
        if "num_workers" in kwargs and kwargs["num_workers"] > 0:
            return AsyncDatasetIterator(self, *args, **kwargs)
        else:
            for key in ["num_workers", "pin_memory", "seed", "mp_start_method"]:
                if key in kwargs:
                    del kwargs[key]
            return DatasetIterator(self, *args, **kwargs)

    def __getattr__(self, name):
        if "dataset" in self.__dict__:
            return getattr(self.__dict__["dataset"], name)
        else:
            raise AttributeError("unknown attribute: %s" % name)

    def get_classy_state(self):
        state = {
            # For debugging saved states
            "state": {"dataset_type": type(self)}
        }
        if "dataset" in self.__dict__:
            state.update(wrapped_state=self.dataset.get_classy_state())

        return state

    def set_classy_state(self, state):
        if "dataset" in self.__dict__:
            self.dataset.set_classy_state(state["wrapped_state"])
        return self
