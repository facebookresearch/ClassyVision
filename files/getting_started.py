#!/usr/bin/env python
# coding: utf-8

# # Getting started with Classy Vision

# Classy Vision is an end-to-end framework for image and video classification. Classy Vision makes it easy to write and launch distributed training jobs.
# 
# In this tutorial, we will cover:
# 1. How to start a new project;
# 2. How to launch a single node training run; 
# 3. How to launch a distributed training run; 
# 4. How to visualize results with Tensorboard; 
# 5. How to load checkpoints and interact with the trained model; 
# 6. How to start training from a Jupyter notebook;
# 7. How to train a ResNet 50 model on ImageNet;
# 
# ## 0. Setup
# 
# Make sure you have Classy Vision installed. To install it, run this in your terminal:

# In[ ]:


! pip install classy_vision

# If you would like to use GPUs for training, make sure your environment has a working version of PyTorch with CUDA:

# In[ ]:


import torch
torch.cuda.is_available()

# The cell above should output `True`. Check out [this link](https://pytorch.org/get-started/locally/) for more details on how to install PyTorch. For this tutorial, we will be using [Tensorboard](https://www.tensorflow.org/tensorboard). Install it with the following (on your terminal):

# In[ ]:


! pip install tensorboard tensorboardX

# ## 1. Start a new project
# 
# To start, let's create a new project. Run this in your terminal:

# In[ ]:


! classy-project my-project

# In[ ]:


%cd my-project

# To launch a training run on the current machine, run the following:

# In[ ]:


!  ./classy_train.py --config configs/template_config.json

# That's it! You've launched your first training run. This trained a small MLP model on a dataset made of random noise, which is not that useful. The `classy-project` utility creates the scaffolding for you project, and you should modify it according to your needs. We'll learn how to customize your runs on the next few tutorials.
# 
# Let's take a look at what `classy-project` has created for us:

# In[ ]:


! find . | grep -v \.pyc | sort

# Here's what each folder means:
# 
#  * `configs`: stores your experiment configurations. Keeping all your experiments as separate configuration files helps making your research reproducible;
#  * `models`: code for your custom model architectures;
#  * `losses`: code for your custom loss functions;
#  * `datasets`: code for your custom datasets;
#  * `classy_train.py`: script to execute a training job; This uses the Classy Vision library to configure the job and execute it, and you might change it according to your needs;
#  * `template_config.json`: experiment configuration file. This file is read by `classy_train.py` to configure your training job and launch it.

# Let's take a peek at the configuration file:

# In[ ]:


! cat configs/template_config.json

# That file can be shared with other researchers whenever you want them to reproduce your experiments. We generate `json` files by default, but `YAML` will be officially supported soon.
# 
# ## 2. Distributed training
# 
# `classy_train.py` can also be called from `torch.distributed.launch`, similar to regular PyTorch distributed scripts:

# In[ ]:


! python -m torch.distributed.launch --use_env --nproc_per_node=2 ./classy_train.py --config configs/template_config.json --distributed_backend ddp

# If you have two GPUs on your current machine, that command will launch one process per GPU and start a [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) training run. 
# 
# ## 3. Tensorboard integration
# 
# [Tensorboard](https://www.tensorflow.org/tensorboard) is a very useful tool for visualizing training progress. Classy Vision works with tensorboard out-of-the-box, just make sure you have it installed as described in the Setup section. By default `classy_train.py` will output tensorboard data in a subdirectory of your project directory (typically named `output_<timestamp>/tensorboard`), so in our case we can just launch tensorboard in the current working directory:

# In[ ]:


%load_ext tensorboard
%tensorboard --logdir .

# You can also customize the tensorboard output directory by editing `classy_train.py`.

# ## 4. Loading checkpoints
# 
# Now that we've run `classy_train.py`, let's see how to load the resulting model. At the end of execution, `classy_train.py` will print the checkpoint directory used for that run. Each run will output to a different directory, typically named `output_<timestamp>/checkpoints`.

# In[ ]:


from classy_vision.generic.util import load_checkpoint
from classy_vision.models import ClassyModel

# This is important: importing models here will register your custom models with Classy Vision
# so that it can instantiate them appropriately from the checkpoint file
# See more information at https://classyvision.ai/api/models.html#classy_vision.models.register_model
import models

# Update this with your actual directory:
checkpoint_dir = './output_<timestamp>/checkpoints'
checkpoint_data = load_checkpoint(checkpoint_dir)
model = ClassyModel.from_checkpoint(checkpoint_data)
model

# That's it! You can now use that model for inference as usual.
# 
# ## 5. Resuming from checkpoints
# 
# Resuming from a checkpoint is as simple as training: `classy_train.py` takes a `--checkpoint_folder` argument, which specifies the checkpoint to resume from:

# In[ ]:


! ./classy_train.py --config configs/template_config.json --checkpoint_folder ./output_<timestamp>/checkpoints

# ## 6. Interactive development
# 
# Training scripts and configuration files are useful for running large training jobs on a GPU cluster (see our [AWS tutorial](https://classyvision.ai/tutorials/ray_aws)), but a lot of day-to-day work happens interactively within Jupyter notebooks. Classy Vision is designed as a library that can be used without our built-in training scripts. Let's take a look at how to do the same training run as before, but within Jupyter instead of using `classy_train.py`:

# In[ ]:


import classy_vision

# In[ ]:


from datasets.my_dataset import MyDataset
from models.my_model import MyModel
from losses.my_loss import MyLoss
from classy_vision.dataset.transforms import GenericImageTransform
from torchvision import transforms

train_dataset = MyDataset(
    batchsize_per_replica=32,
    shuffle=False,
    transform=GenericImageTransform(
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    ),
    num_samples=100,
    crop_size=224,
    class_ratio=0.5,
    seed=0,
)

test_dataset = MyDataset(
    batchsize_per_replica=32,
    shuffle=False,
    transform=GenericImageTransform(
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    ),
    num_samples=100,
    crop_size=224,
    class_ratio=0.5,
    seed=0,
)


# In[ ]:


from classy_vision.tasks import ClassificationTask
from classy_vision.optim import SGD
from classy_vision.optim.param_scheduler import LinearParamScheduler

model = MyModel()
loss = MyLoss()

optimizer = SGD(momentum=0.9, weight_decay=1e-4, nesterov=True)
optimizer.set_param_schedulers(
    {"lr": LinearParamScheduler(start_lr=0.01, end_lr=0.009)}
)

from classy_vision.trainer import LocalTrainer

task = ClassificationTask() \
        .set_model(model) \
        .set_dataset(train_dataset, "train") \
        .set_dataset(test_dataset, "test") \
        .set_loss(loss) \
        .set_optimizer(optimizer) \
        .set_num_epochs(1)

trainer = LocalTrainer()
trainer.train(task)

# That's it! Your model is trained now and ready for inference:

# In[ ]:


import torch
x = torch.randn((1, 3, 224, 224))
with torch.no_grad():
    y_hat = model(x)

y_hat

# ## 7. Training a ResNet 50 on ImageNet
# 
# We have looked at training models using synthetic data so far. A more typical workflow involves training a model on a real world dataset like [ImageNet](http://image-net.org/), which we will cover in this section.
# 
# To be able to train using ImageNet, first download the dataset archives from http://image-net.org/. Then, extract the data to a format expected by [`torchvision.datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) inside subdirectories for the individual splits (`train` and `val`). We can then pass the root path containing these archives to the [`ImageNetDataset`](https://classyvision.ai/api/dataset.html#classy_vision.dataset.ImageNetDataset).
# 
# The following configuration can be used to train a ResNet 50 on ImageNet to `76.4%` top-1 accuracy in 90 epochs. The optimizer configuration uses SGD with momentum, gradual learning rate warm up for the first 5 epochs and 1/10 learning rate drops at epochs 30, 60 and 80. The learning rate is calculated for a setup with 32 GPUs and can be scaled based on the overall batch size [1].

# In[4]:


config = {
    "name": "classification_task",
    "num_epochs": 90,
    "loss": {
        "name": "CrossEntropyLoss"
    },
    "dataset": {
        "train": {
            "name": "classy_imagenet",
            "split": "train",
            "batchsize_per_replica": 32,
            "num_samples": None,
            "use_shuffle": True,
            "root": "/path/to/imagenet/"  # replace with path to the extracted dataset
        },
        "test": {
            "name": "classy_imagenet",
            "split": "val",
            "batchsize_per_replica": 32,
            "num_samples": None,
            "use_shuffle": False,
            "root": "/path/to/imagenet/"  # replace with path to the extracted dataset
        }
    },
    "meters": {
        "accuracy": {
            "topk": [1, 5]
        }
    },
    "model": {
        "name": "resnet",
        "num_blocks": [3, 4, 6, 3],
        "small_input": False,
        "zero_init_bn_residuals": True,
        "heads": [
          {
            "name": "fully_connected",
            "unique_id": "default_head",
            "num_classes": 1000,
            "fork_block": "block3-2",
            "in_plane": 2048
          }
        ]
    },
    "optimizer": {
        "name": "sgd",
        "param_schedulers": {
            "lr": {
                "name": "composite",
                "schedulers": [
                    {"name": "linear", "start_lr": 0.1, "end_lr": 0.4},
                    {"name": "multistep", "values": [0.4, 0.04, 0.004, 0.0004], "milestones": [30, 60, 80]}
                ],
                "update_interval": "epoch",
                "interval_scaling": ["rescaled", "fixed"],
                "lengths": [0.0555, 0.9445]
            }
        },
        "weight_decay": 1e-4,
        "momentum": 0.9
    }
}

# ## 8. Conclusion
# 
# In this tutorial, we learned how to start a new project using Classy Vision, how to perform tranining locally and how to do multi-gpu training on a single machine. We also saw how to use Tensorboard to visualize training progress, how to load models from checkpoints and how resume training from a checkpoint file. We also went over how to use the ImageNet dataset to train a ResNet 50. In the next tutorials, we'll look into how to add custom datasets, models and loss functions to Classy Vision so you can adapt it to your needs, and how to launch distributed training on multiple nodes.

# ## 9. References
# 
# [1] Goyal, Priya, et al. "Accurate, large minibatch sgd: Training imagenet in 1 hour." arXiv preprint arXiv:1706.02677 (2017).

# In[ ]:



