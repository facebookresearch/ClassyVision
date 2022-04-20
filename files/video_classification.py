#!/usr/bin/env python
# coding: utf-8

# # How to do video classification 

# In this tutorial, we will show how to train a video classification model in Classy Vision. Given an input video, the video classification task is to predict the most probable class label. This is very similar to image classification, which was covered in other tutorials, but there are a few differences that make video special. As the video duration can be long, we sample short video clips of a small number of frames, use the classifier to make predictions, and finally average the clip-level predictions to get the final video-level predictions. 
# 
# In this tutorial we will: (1) load a video dataset; (2) configure a video model; (3) configure video meters; (4) build a task; (5) start training; Please note that these steps are being done separately in the tutorial for easy of exposition in the notebook format. As described in our [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial, you can combine all configs used in this tutorial into a single config for ClassificationTask and train it using `classy_train.py`.

# Before we get started, let us enable INFO level logging so that we can monitor the progress of our runs.

# In[ ]:


import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ## 1. Prepare the dataset
# 
# All right! Let's start with the dataset. [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) is a canonical action recognition dataset. It has 101 action classes, and has 3 folds with different training/testing splitting . We use fold 1 in this tutorial. Classy Vision has implemented the dataset `ucf101`, which can be used to load the training and testing splits. 

# ### 1.1 Directories and Metadata File information
# 
# You will need to download the videos and the split files of UCF-101 dataset from the [official site](https://www.crcv.ucf.edu/data/UCF101.php). 
# 
# You should then have the videos present in a folder -
# 
# ```console
# $ ls /path/to/ucf101
# ApplyEyeMakeup
# ...
# YoYo
# ```
# 
# There also needs to be a folder which contains the split files of the dataset -
# 
# ```console
# $ ls /path/to/UCF101TrainTestSplits-RecognitionTask
# classInd.txt
# ...
# trainlist03.txt
# ```
# 
# Upon initializing the dataset, Classy Vision processes all this dataset information and stores it in a metadata file. This metadata file can be re-used for future runs to make the initialization faster. You can pass the path to store the metadata as `/path/to/ucf101_metadata.pt`.

# In[ ]:


from classy_vision.dataset import build_dataset

# set it to the folder where video files are saved
video_dir = "/path/to/ucf101"
# set it to the folder where dataset splitting files are saved
splits_dir = "/path/to/UCF101TrainTestSplits-RecognitionTask"
# set it to the file path for saving the metadata
metadata_file = "/path/to/ucf101_metadata.pt"

datasets = {}
datasets["train"] = build_dataset({
    "name": "ucf101",
    "split": "train",
    "batchsize_per_replica": 8,  # For training, we use 8 clips in a minibatch in each model replica
    "use_shuffle": True,         # We shuffle the clips in the training split
    "num_samples": 64,           # We train on 16 clips in one training epoch
    "clips_per_video": 1,        # For training, we randomly sample 1 clip from each video
    "frames_per_clip": 8,        # The video clip contains 8 frames
    "video_dir": video_dir,
    "splits_dir": splits_dir,
    "metadata_file": metadata_file,
    "fold": 1,
    "transforms": {
        "video": [
            {
                "name": "video_default_augment",
                "crop_size": 112,
                "size_range": [128, 160]
            }
        ]
    }
})
datasets["test"] = build_dataset({
    "name": "ucf101",
    "split": "test",
    "batchsize_per_replica": 10,  # For testing, we will take 1 video once a time, and sample 10 clips per video
    "use_shuffle": False,         # We do not shuffle clips in the testing split
    "num_samples": 80,            # We test on 80 clips in one testing epoch
    "clips_per_video": 10,        # We sample 10 clips per video
    "frames_per_clip": 8,
    "video_dir": video_dir,
    "splits_dir": splits_dir,
    "metadata_file": metadata_file,
    "fold": 1,
    "transforms": {
        "video": [
            {
                "name": "video_default_no_augment",
                "size": 128
            }
        ]
    }    
})

# Note we specify different transforms for training and testing split. For training split, we first randomly select a size from `size_range` [128, 160], and resize the video clip so that its short edge is equal to the random size. After that, we take a random crop of spatial size 112 x 112. We find such data augmentation helps the model generalize better, and use it as the default transform with data augmentation. For testing split, we resize the video clip to have short edge of size 128, and skip the random cropping to use the entire video clip. This is the default transform without data augmentation.

# ## 2. Define a model trunk and a head
# 
# Next, let's create the video model, which consists of a trunk and a head. The trunk can be viewed as a feature extractor for computing discriminative features from raw video pixels while the head is viewed as a classifier for producing the final predictions. Let's first create the trunk of architecture ResNet3D-18 by using the built-in `resnext3d` model in Classy Vision.

# In[ ]:


from classy_vision.models import build_model

model = build_model({
    "name": "resnext3d",
    "frames_per_clip": 8,        # The number of frames we have in each video clip
    "input_planes": 3,           # We use RGB video frames. So the input planes is 3
    "clip_crop_size": 112,       # We take croppings of size 112 x 112 from the video frames 
    "skip_transformation_type": "postactivated_shortcut",    # The type of skip connection in residual unit
    "residual_transformation_type": "basic_transformation",  # The type of residual connection in residual unit
    "num_blocks": [2, 2, 2, 2],  # The number of residual blocks in each of the 4 stages 
    "input_key": "video",        # The key used to index into the model input of dict type 
    "stage_planes": 64,    
    "num_classes": 101           # the number of classes
})

# We also need to create a model head, which consists of an average pooling layer and a linear layer, by using the `fully_convolutional_linear` head. At test time, the shape (channels, frames, height, width) of input tensor is typically `(3 x 8 x 128 x 173)`. The shape of input tensor to the average pooling layer is `(512, 1, 8, 10)`. Since we do not use a global average pooling but an average pooling layer of kernel size `(1, 7, 7)`, the pooled feature map has shape `(512, 1, 2, 5)`. The shape of prediction tensor from the linear layer is `(1, 2, 5, 101)`, which indicates the model computes a 101-D prediction vector densely over a `2 x 5` grid. That's why we name the head as `FullyConvolutionalLinearHead` because we use the linear layer as a `1x1` convolution layer to produce spatially dense predictions. Finally, predictions over the `2 x 5` grid are averaged.

# In[ ]:


from classy_vision.heads import build_head
from collections import defaultdict

unique_id = "default_head"
head = build_head({
    "name": "fully_convolutional_linear",
    "unique_id": unique_id,
    "pool_size": [1, 7, 7],
    "num_classes": 101,
    "in_plane": 512    
})
# In Classy Vision, the head can be attached to any residual block in the trunk. 
# Here we attach the head to the last block as in the standard ResNet model
fork_block = "pathway0-stage4-block1"
heads = {fork_block: [head]}
model.set_heads(heads)

# ## 3. Choose the meters
# 
# This is the biggest difference between video and image classification. For images we used `AccuracyMeter` to measure top-1 and top-5 accuracy. For videos you can also use both `AccuracyMeter` and `VideoAccuracyMeter`, but they behave differently:
#  * `AccuracyMeter` takes one clip-level prediction and compare it with groundtruth video label. It reports the clip-level accuracy.
#  * `VideoAccuracyMeter` takes multiple clip-level predictions from the same video, averages them and compares that with groundtruth video label. It reports the video-level accuracy which is usually higher than clip-level accuracy. 
#  
#  Both meters report top-1 and top-5 accuracy.

# In[ ]:


from classy_vision.meters import build_meters, AccuracyMeter, VideoAccuracyMeter

meters = build_meters({
    "accuracy": {
        "topk": [1, 5]
    },
    "video_accuracy": {
        "topk": [1, 5],
        "clips_per_video_train": 1,
        "clips_per_video_test": 10
    }
})

# ## 4. Build a task
# Great! we have defined the minimal set of components necessary for video classification, including dataset, model, loss function, meters and optimizer. We proceed to define a video classification task, and populate it with all the components.

# In[ ]:


from classy_vision.tasks import ClassificationTask
from classy_vision.optim import build_optimizer
from classy_vision.losses import build_loss

loss = build_loss({"name": "CrossEntropyLoss"})

optimizer = build_optimizer({
    "name": "sgd",
    "param_schedulers": {
        "lr": {
            "name": "multistep",
            "values": [0.005, 0.0005],
            "milestones": [1]
        }
    },
    "num_epochs": 2,
    "weight_decay": 0.0001,
    "momentum": 0.9
})

num_epochs = 2
task = (
    ClassificationTask()
    .set_num_epochs(num_epochs)
    .set_loss(loss)
    .set_model(model)
    .set_optimizer(optimizer)
    .set_meters(meters)
) 
for phase in ["train", "test"]:
    task.set_dataset(datasets[phase], phase)

# ## 5. Start training
# 
# After creating a task, you can simply pass that to a Trainer to start training. Here we will train on a single node and 
# configure logging and checkpoints for training:

# In[ ]:


import time
import os

from classy_vision.trainer import LocalTrainer
from classy_vision.hooks import CheckpointHook
from classy_vision.hooks import LossLrMeterLoggingHook

hooks = [LossLrMeterLoggingHook(log_freq=4)]

checkpoint_dir = f"/tmp/classy_checkpoint_{time.time()}"
os.mkdir(checkpoint_dir)
hooks.append(CheckpointHook(checkpoint_dir, input_args={}))

task = task.set_hooks(hooks)

trainer = LocalTrainer()
trainer.train(task)

# As the training progresses, you should see `LossLrMeterLoggingHook` printing the loss, learning rate and meter metrics. Checkpoints will be available in the folder created above.
# 
# ## 6. Conclusion
# 
# Video classification is very similar to image classification in Classy Vision, you just need to use an appropriate dataset, model and meters. This tutorial glossed over many details about training, please take a look at our [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial to learn more. Refer to our API reference for more details about [ResNeXt3D](https://classyvision.ai/api/models.html#classy_vision.models.ResNeXt3D) models, [UCF101](https://classyvision.ai/api/dataset.html#classy_vision.dataset.UCF101Dataset) dataset and [VideoAccuracy](http://classyvision.ai/api/meters.html#classy_vision.meters.VideoAccuracyMeter) meters.
# 
