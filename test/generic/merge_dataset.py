#!/usr/bin/env python3

from classy_vision.dataset.core.dataset import Dataset


class MergeDataset(Dataset):
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
        assert all(isinstance(dataset, Dataset) for dataset in datasets)
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

    def get_classy_state(self):
        state = {"state": {"dataset_type": type(self)}}
        if isinstance(self.datasets, list):
            state["wrapped_states"] = [
                dataset.get_classy_state() for dataset in self.datasets
            ]
        else:
            state["wrapped_states"] = {
                key: dataset.get_classy_state()
                for key, dataset in self.datasets.items()
            }

        return state

    def set_classy_state(self, state):
        if isinstance(self.datasets, list):
            for idx, dataset in enumerate(self.datasets):
                dataset.set_classy_state(state["wrapped_states"][idx])
        else:
            for key, dataset in self.datasets.items():
                dataset.set_classy_state(state["wrapped_states"][key])

        return self
