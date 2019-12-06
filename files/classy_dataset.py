#!/usr/bin/env python
# coding: utf-8

# # Creating a custom dataset

# In this tutorial we will learn how to do the following: 
# 
# 1. Create a custom dataset within Classy Vision
# 2. Integrate a dataset with Classy Vision's configuration system
# 3. Iterate over the samples contained in a dataset
# 4. Using transforms with Classy Vision
# 5. Create a ImageNet dataset, using standard transforms, using torchvision
# 
# If you haven't already read our [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial, we recommend starting there before reading this tutorial.

# ## 1. Create a custom dataset within Classy Vision
# 
# Creating a dataset for use / using an existing dataset in Classy Vision is as easy as it is in PyTorch, it only requires wrapping the dataset in our dataloading class, ClassyDataset.
# 
# First, specify a dataset with a `__getitem__` and `__len__` function, the same as required by torch.utils.data.Dataset

# In[ ]:


import torch.utils.data
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.length = 100
        
    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.length, \
            "Provided index {} must be in range [0, {}).".format(idx, self.length)
        return torch.rand(3, 100, 100)
    
    def __len__(self):
        return self.length

# Now for most training tasks we want to be able to configure the batchsize on the fly, transform samples, shuffle the dataset, maybe limit the number of samples to shorten a training run, and then construct an iterator for the training loop. ClassyDataset is a simple wrapper that provides this functionality.   

# In[ ]:


from classy_vision.dataset import ClassyDataset

class MyClassyDataset(ClassyDataset):
    def __init__(self, split, batchsize_per_replica, shuffle, transform, num_samples):
        dataset = MyDataset()
        super().__init__(dataset, split, batchsize_per_replica, shuffle, transform, num_samples)

# It's that easy! Later in the tutorial we will see how to use the iterator, but before moving on, let's talk about what each of these arguments does.
# - __split__: is a string specifying the split of the data, typically either "train" or "test". This is optional, not needed for many datasets.
# - __batchsize_per_replica__: the batchsize per trainer (so if you have 8 GPUs with 1 trainer processes and a batchsize_per_replica of 32, then your batchsize for single update is 8 * 32 = 256).
# - __shuffle__: If true, then shuffle the dataset before each epoch.
# - __transform__: A callable applied to each sample before returning. Note that this can get tricky since many datasets (e.g. torchvision datasets) return complex samples containing both the image / video content and a label and possibly additional metadata. We pass the _whole_ sample to the transform, so it needs to know how to parse the sample...more on this later.
# - __num_samples__: Not needed in the standard use cases, but this allows a user to adjust the length of samples retrieved in an epoch, can be convenient for debugging via config (e.g. setting num_samples = 10 will speed up training). By default this is set to None and iteration proceeds over the whole dataset.
# 
# To get started with a basic task just do:

# In[ ]:


from classy_vision.tasks import ClassificationTask

my_dataset = MyClassyDataset(
    split="train", 
    batchsize_per_replica=10, 
    shuffle=True, 
    transform=None, 
    num_samples=None,
)

# Note, the "train" here is the phase type, which is unrelated to the split name.
# It tells the task to set the model in train mode / do a backwards pass, etc using
# this dataset...the split argument helps the dataset decide which training data to load.
my_task = ClassificationTask().set_dataset(my_dataset, "train")

# For more details on training a model, please see our [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial.

# ## 2. Integrating a dataset with Classy Vision's configuration system
# 
# Classy Vision is also able to read a configuration file and instantiate the dataset. This is useful to keep your experiments organized and reproducible. For that, you have to:
# 
# - Implement a from_config method
# - Add the register_model decorator to MyClassyDataset

# In[ ]:


from classy_vision.dataset import ClassyDataset, register_dataset
from classy_vision.dataset.transforms import build_transforms

@register_dataset("my_dataset")
class MyClassyDataset(ClassyDataset):
    def __init__(self, split, batchsize_per_replica, shuffle, transform, num_samples):
        dataset = MyDataset()
        super().__init__(dataset, split, batchsize_per_replica, shuffle, transform, num_samples)
        
    @classmethod
    def from_config(cls, config):
        transform = build_transforms(config["transforms"])
        return cls(
            split=config["split"],
            batchsize_per_replica=config["batchsize_per_replica"],
            shuffle=config["shuffle"],
            transform=transform,
            num_samples=config["num_samples"],
        )

# Now we can start using this dataset in our configurations. The string argument passed to the register_dataset is a unique identifier for this model (if you try to register two models with the same name, it will throw an error):
# 

# In[ ]:


from classy_vision.dataset import build_dataset
import torch

dataset_config = {
    "name": "my_dataset",
    "split": "train",
    "batchsize_per_replica": 10,
    "shuffle": True,
    "transforms": [{"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}],
    "num_samples": None,
}
my_dataset = build_dataset(dataset_config)
assert isinstance(my_dataset, MyClassyDataset)

sample = my_dataset[0]
print(sample.size())

# ## 3. Iterate over the samples contained in a dataset
# 
# As mentioned above, the ClassyDataset class adds several pieces of basic logic for constructing a torch.utils.data.Dataloader for your dataset. ClassyDataset supports local and distributed training out-of-box by internally using a PyTorch DistributedSampler for sampling the dataset along with the PyTorch Dataloader for batching and parallelizing sample retrieval. To get an iterable for epoch 0, do the following:

# In[ ]:


from classy_vision.dataset import build_dataset
import torch

dataset_config = {
    "name": "my_dataset",
    "split": "train",
    "batchsize_per_replica": 10,
    "shuffle": True,
    "transforms": [],
    "num_samples": None,
}
my_dataset = build_dataset(dataset_config)
assert isinstance(my_dataset, MyClassyDataset)

# multiprocessing_context can be set to "spawn", "forkserver", "fork" or None.
# If None is used, then the dataloader inherits the context of the parent thread.
# If num_workers is 0, then multiprocessing is not used by the dataloader
#
# A warning, while fork is fast and simple to get started with, it 
# is unsafe to use with threading and can lead to difficult to debug errors.
# Spawn / forkserver are threadsafe, but they come with additional startup costs.
iterator = my_dataset.iterator(
    shuffle_seed=0,
    epoch=0,
    num_workers=0,  # 0 indicates to do dataloading on the master process
    pin_memory=False,
    multiprocessing_context=None,
)
assert isinstance(iterator, torch.utils.data.DataLoader)

# Iterate over all 100 samples.
for sample in iter(iterator):
    # Do stuff with sample...
    # Note that size now has an extra dimension representing the batchsize
    assert sample.size() == torch.Size([10, 3, 100, 100])

# You can also provide a custom iterator function if you would like to return a custom iterator or a custom sampler. Please see the ClassyDataset code for more details.

# ## 4. Using transforms with Classy Vision
# 
# You may have noticed in the configuration section that we did something slightly more complicated with the transform configuration. In particular, just like our datasets / models etc, we have a registration / build mechanism for transforms so that transforms can be specified via config. 

# #### Transform example using Classy Vision's synthetic image dataset
# We also automatically register torchvision transforms, so let's start with an example of how to specify torchvision transforms and the synthetic image dataset we provide for testing / proto-typing.
# 

# In[ ]:


import torchvision.transforms as transforms
from classy_vision.dataset import build_dataset
from classy_vision.dataset.classy_synthetic_image import SyntheticImageDataset
from classy_vision.dataset.transforms import build_transforms

# Declarative approach

# Transform to be applied to image
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

decl_dataset = SyntheticImageDataset(
    batchsize_per_replica=10,
    shuffle=True,
    transform=image_transform,
    num_samples=100,
    crop_size=320,
    class_ratio=4,
    seed=0,
)

# FAILS!!!!
# decl_dataset[0]

# This fails! Why?
# 
# It fails because most datasets don't return just an image, they return image or video content data, label data, and (potentially) sample metadata. In Classy Vision, the sample format is specified by the task and our classification_task expects a dict with input / target keys.
# 
# For example, the sample format for the SyntheticImageDataset looks like:

# `{"input": <PIL Image>, "target": <Target>}`

# For our transforms to work, we need to specify which key to apply the transform to.

# In[ ]:


import torchvision.transforms as transforms
from classy_vision.dataset import build_dataset
from classy_vision.dataset.classy_synthetic_image import SyntheticImageDataset
from classy_vision.dataset.transforms import build_transforms, ApplyTransformToKey

# Declarative approach

# Transform to be applied to image
image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transform wrapper that says which key to apply the transform to
transform = ApplyTransformToKey(
    transform=image_transform,
    key="input",
)

decl_dataset = SyntheticImageDataset(
    batchsize_per_replica=10,
    shuffle=True,
    transform=transform,
    num_samples=100,
    crop_size=320,
    class_ratio=4,
    seed=0,
)

# Success!!!!
decl_dataset[0]

# Now let's see how to do the same thing via a config.

# In[ ]:


# Note that this cell won't work until we fix the synthetic dataset from_config function

from classy_vision.dataset import build_dataset

# Configuration approach
config = {
    "name": "synthetic_image",
    "batchsize_per_replica": 10,
    "use_shuffle": True,
    "transforms": [
        {
            "name": "apply_transform_to_key",
            "transforms": [
                {"name": "Resize", "size": 256},
                {"name": "CenterCrop", "size": 224},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ],
            "key": "input",
        },
    ],
    "num_samples": 100,
    "crop_size": 320,
    "class_ratio": 4,
    "seed": 0
}

config_dataset = build_dataset(config)

# Sample should be the same as that provided by the decl_dataset
assert torch.allclose(config_dataset[0]["input"], decl_dataset[0]["input"])

# #### Transform example for a torchvision dataset
# Torchvision has a different sample format using tuples for images: 
# 
# `(<PIL Image>, <Target>)`
# 
# The ApplyTransformToKey will still work (the key in this case is '0'), but for our classification tasks, we also want a sample that is a dict with "input"/"target" keys. 
# 
# Because this is a common dataset format, we provide a convenience transform called "GenericImageTransform" which applies a specified transform to the torchvision tuple image key and then maps the whole sample to a dict. This is just a convenience transform, we can also do this using raw composable blocks, but it makes things more verbose.
# 
# All of the transforms in the next cell have the same effect on an image:

# In[ ]:


from torchvision.transforms import Compose
from classy_vision.dataset.transforms import build_transforms
from classy_vision.dataset.transforms.util import GenericImageTransform

# Declarative
image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
decl_transform = GenericImageTransform(transform=image_transform)

# Configuration with helper function
transform_config = [{
    "name": "generic_image_transform",
    "transforms": [
        {"name": "Resize", "size": 256},
        {"name": "CenterCrop", "size": 224},
        {"name": "ToTensor"},
        {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    ], 
}]
config_helper_transform = build_transforms(transform_config)

# Configuration using raw, composable functions:
transform_config = [
    {"name": "tuple_to_map", "list_of_map_keys": ["input", "target"]},
    {
        "name": "apply_transform_to_key",
        "transforms": [
            {"name": "Resize", "size": 256},
            {"name": "CenterCrop", "size": 224},
            {"name": "ToTensor"},
            {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        ], 
        "key": "input",
    },
]
config_raw_transform = build_transforms(transform_config)

# These transforms are all functionally the same

# ## 5. Create a Classy Imagenet
# 
# Now, to complete this tutorial, we show our code for creating an ImageNet dataset in classy vision using the pre-existing torchvision dataset. Code very similar to this (+ some typing and helper functions) is in the datasets folder of the base Classy Vision repository.
# 
# Note, we do not distribute any of the underlying dataset data with Classy Vision. Before this will work, you will need to download a torchvision compatible copy of the Imagenet dataset yourself.

# In[ ]:


from classy_vision.dataset import ClassyDataset, register_dataset
from classy_vision.dataset.transforms import ClassyTransform, build_transforms
from torchvision.datasets.imagenet import ImageNet
        
        
@register_dataset("example_imagenet")
class ExampleImageNetDataset(ClassyDataset):
    def __init__(
        self,
        split,
        batchsize_per_replica,
        shuffle,
        transform,
        num_samples,
        root,  # Root directory for your Imagenet dataset
    ):  
        # Create torchvision dataset
        dataset = ImageNet(root=root, split=split)
        super().__init__(
            dataset, split, batchsize_per_replica, shuffle, transform, num_samples
        )   

    @classmethod
    def from_config(cls, config):
        batchsize_per_replica = config.get("batchsize_per_replica")
        shuffle = config.get("use_shuffle")
        num_samples = config.get("num_samples")
        transform_config = config.get("transforms")
        split = config.get("split")
        root = config.get("root")
        download = config.get("download")
        
        transform = build_transforms(transform_config)
        return cls(
            split=split,
            batchsize_per_replica=batchsize_per_replica,
            shuffle=shuffle,
            transform=transform,
            num_samples=num_samples,
            root=root,
            download=download,
        )

# ## Conclusion
# In this tutorial we have seen how to create a custom dataset using ClassyDataset, how to integrate this dataset with the configuration system, how to iterate over samples / use multiple workers, how to use transforms in the configuration system and finally we showed an example of how to use a torchvision dataset in Classy Vision. 
# 
# For more details on how to use the dataset for training, please see [Getting started](https://classyvision.ai/tutorials/getting_started).
