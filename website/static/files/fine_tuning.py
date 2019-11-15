#!/usr/bin/env python
# coding: utf-8

# #### Let us begin by pre-training a model using a head with 1000 classes

# We want to train for 4 epochs.

# In[6]:


num_epochs = 4


# We will be using synthetic train and test datasets for this example.

# In[7]:


from classy_vision.dataset import SyntheticImageClassificationDataset

train_dataset = SyntheticImageClassificationDataset.from_config({
    "batchsize_per_replica": 32,
    "num_samples": 2000,
    "crop_size": 224,
    "class_ratio": 0.5,
    "seed": 0,
    "use_shuffle": True,
    "split": "train",
})
test_dataset = SyntheticImageClassificationDataset.from_config({
    "batchsize_per_replica": 32,
    "num_samples": 200,
    "crop_size": 224,
    "class_ratio": 0.5,
    "seed": 0,
    "use_shuffle": False,
    "split": "test",
})


# Let us create a ResNet 50 model now.

# In[8]:


from classy_vision.models import ResNet

model = ResNet.from_config({
    "num_blocks": [3, 4, 6, 3],
    "small_input": False,
    "zero_init_bn_residuals": True
})


# Now, we will create a head with 1000 classes.

# In[9]:


from classy_vision.heads import FullyConnectedHead

head = FullyConnectedHead(unique_id="default_head", num_classes=1000, in_plane=2048)


# Let us attach the head to the final block of the model.
# 
# For ResNet 50, we want to attach to the `3`<sup>rd</sup> block in the `4`<sup>th</sup> layer (based on `[3, 4, 6, 3]`). The blocks use 0 indexing, so this maps to `"block3-2"`.

# In[10]:


model.set_heads({"block3-2": {head.unique_id: head}})


# We can use a cross entropy loss from Pytorch.

# In[11]:


from torch.nn.modules.loss import CrossEntropyLoss

loss = CrossEntropyLoss()


# For the optimizer, we will be using SGD.

# In[12]:


from classy_vision.optim import build_optimizer


optimizer = build_optimizer({
    "name": "sgd",
    "lr": {"name": "step", "values": [0.1, 0.01]},
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "num_epochs": num_epochs
})


# We want to track the top-1 and top-5 accuracies of the model.

# In[13]:


from classy_vision.meters import AccuracyMeter

meters = [AccuracyMeter(topk=[1, 5])]


# Let's create a directory to save the checkpoints.

# In[14]:


import os
import time

pretrain_checkpoint_dir = f"/tmp/checkpoint_{time.time()}"
os.mkdir(pretrain_checkpoint_dir)


# Add `ProgressBarHook` to monitor the progress, `LossLrMeterLoggingHook` to monitor the loss and `CheckpointHook` to save the checkpoints.

# In[15]:


from classy_vision.hooks import CheckpointHook, LossLrMeterLoggingHook, ProgressBarHook

hooks = [
    ProgressBarHook(),
    LossLrMeterLoggingHook(),
    CheckpointHook(pretrain_checkpoint_dir, input_args={})
]


# We have all the components ready to setup our pre-training task which trains for 4 epochs.

# In[16]:


from classy_vision.tasks import ClassificationTask

pretrain_task = (
    ClassificationTask()
    .set_num_epochs(num_epochs)
    .set_loss(loss)
    .set_model(model)
    .set_optimizer(optimizer)
    .set_meters(meters)
    .set_hooks(hooks)
    .set_dataset(train_dataset, "train")
    .set_dataset(test_dataset, "test")
)


# Let us train using a local trainer instance.

# In[17]:


from classy_vision.trainer import LocalTrainer

trainer = LocalTrainer()


# Now, we can start training!

# In[18]:


trainer.train(pretrain_task)


# Training is done! Let us now load the saved checkpoint, we will use this later for fine tuning.

# In[19]:


from classy_vision.generic.util import load_checkpoint

pretrained_checkpoint = load_checkpoint(pretrain_checkpoint_dir)


# #### Now we will fine tune a model using a head with 2 classes

# We only want to train our fine tuning task for 1 just epoch.

# In[20]:


num_epochs = 1


# We can re-use the same synthetic datasets as before.

# Let us again create a ResNet 50 model.

# In[21]:


from classy_vision.models import ResNet

model = ResNet.from_config({
    "num_blocks": [3, 4, 6, 3],
    "small_input": False,
    "zero_init_bn_residuals": True
})


# For fine tuning, we will create a head with just 2 classes

# In[22]:


from classy_vision.heads import FullyConnectedHead

head = FullyConnectedHead(unique_id="default_head", num_classes=2, in_plane=2048)


# Let us attach the head to the final block of the model, like before.

# In[23]:


model.set_heads({"block3-2": {head.unique_id: head}})


# For the optimizer, we will be using RMSProp this time.

# In[24]:


from classy_vision.optim import build_optimizer


optimizer = build_optimizer({
    "name": "rmsprop",
    "lr": {"name": "step", "values": [0.1, 0.01]},
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "alpha": 0.9,
    "eps": 1e-3,
    "num_epochs": num_epochs
})


# We want to track the top-1 accuracy of the model.

# In[27]:


from classy_vision.meters import AccuracyMeter

meters = [AccuracyMeter(topk=[1])]


# We will create a new directory to save the checkpoints for our fine tuning run.

# In[28]:


import os
import time

fine_tuning_checkpoint_dir = f"/tmp/checkpoint_{time.time()}"
os.mkdir(fine_tuning_checkpoint_dir)


# Hooks are also the same as before.

# In[29]:


from classy_vision.hooks import CheckpointHook, LossLrMeterLoggingHook, ProgressBarHook

hooks = [
    ProgressBarHook(),
    LossLrMeterLoggingHook(),
    CheckpointHook(fine_tuning_checkpoint_dir, input_args={})
]


# Now we can setup our fine tuning task.

# In[30]:


from classy_vision.tasks import FineTuningTask

fine_tuning_task = (
    FineTuningTask()
    .set_num_epochs(num_epochs)
    .set_loss(loss)
    .set_model(model)
    .set_optimizer(optimizer)
    .set_meters(meters)
    .set_hooks(hooks)
    .set_dataset(train_dataset, "train")
    .set_dataset(test_dataset, "test")
)


# Since this is a fine tuning task, there are some other configurations which need to be done.

# We don't want to re-train the trunk, so we will be freezing it. This is optional.

# In[31]:


fine_tuning_task.set_freeze_trunk(True)


# We want to start training the heads from scratch, so we will be resetting them. This is required in this example since the pre-trained heads are not compatible with the heads in fine tuning (they have different number of classes). Otherwise, this is also optional.

# In[32]:


fine_tuning_task.set_reset_heads(True)


# We need to give our task the pre-trained checkpoint, which it'll need to start pre-training on.

# In[33]:


fine_tuning_task.set_pretrained_checkpoint(pretrained_checkpoint)


# Let us fine tune!

# In[34]:


trainer.train(fine_tuning_task)

