Each dataset implements a single function `get_dataset(split)`, where `split` can
have the values `train` and `test`. It returns a tuple containing the torch
Dataset objects, and the number of classes in the dataset.
