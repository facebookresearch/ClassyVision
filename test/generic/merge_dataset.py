#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class MergeDataset:
    """
        Dataset that merges samples from multiple datasets into single sample.

        If datasets have distinct keys, then we merge dicts, e.g.

            dataset1[idx] = {'input': input_tensor}
            dataset2[idx] = {'target': target_tensor}
            merged_dataset[idx] = {'input': input_tensor, 'target': target_tensor}

        If datasets have matching keys then we create a list and append, e.g.

            dataset1[idx] = {'input': input_tensor1}
            dataset2[idx] = {'input': input_tensor2}
            merged_dataset[idx] = {'input': [input_tensor1, input_tensor2]}

        Note, if your datasets' samples do not have consistent keys for each sample,
        this could lead to inconsistent samples merged samples.
    """

    def __init__(self, datasets):

        # assertions:
        assert isinstance(datasets, list)
        assert all(len(dataset) == len(datasets[0]) for dataset in datasets)

        # create object:
        super(MergeDataset, self).__init__()
        self.datasets = datasets

    def __getitem__(self, idx):
        final_sample = {}
        for dataset in self.datasets:
            curr_sample = dataset[idx]
            assert isinstance(curr_sample, dict), "Merge dataset only supports dicts"
            for key in curr_sample.keys():
                # If keys are distinct, then
                if key not in final_sample:
                    final_sample[key] = curr_sample[key]
                elif not isinstance(final_sample[key], list):
                    final_sample[key] = [final_sample[key], curr_sample[key]]
                else:
                    final_sample[key].append(curr_sample[key])

        return final_sample

    def __len__(self):
        return len(self.datasets[0])
