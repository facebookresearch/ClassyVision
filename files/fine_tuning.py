#!/usr/bin/env python
# coding: utf-8

# # Fine tuning a model
# 
# Fine tuning is a form of transfer learning: when you only have a small labeled dataset for a specific task, you can pick up a model trained for a different task and fine-tune it for your specific dataset. These pre-trained models are usually trained on much larger datasets, which helps improving performance. 
# 
# In this tutorial we'll look into how to pick up a pre-trained model and fine tune it for a different task. In part (1) we'll train a model and save it to a checkpoint file. In part (2), we'll load the checkpoint file and run the fine-tuning. Feel free to skip part (1) if you already have a checkpoint file to begin with.

# ## 1. Training a model
# Let us begin by pre-training a model using a head with 1000 classes

# We want to train for 4 epochs.

# In[ ]:


num_epochs = 4


# We will be using synthetic train and test datasets for this example. The transforms used are from torchvision and are applied to the input value in the sample (rather than the target).

# In[1]:


from classy_vision.dataset import SyntheticImageDataset

train_dataset = SyntheticImageDataset.from_config({
    "batchsize_per_replica": 32,
    "num_samples": 2000,
    "crop_size": 224,
    "class_ratio": 0.5,
    "seed": 0,
    "use_shuffle": True,
    "transforms": [{
        "name": "apply_transform_to_key",
        "transforms": [
            {"name": "ToTensor"},
            {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        ],
        "key": "input"
    }]
})
test_dataset = SyntheticImageDataset.from_config({
    "batchsize_per_replica": 32,
    "num_samples": 200,
    "crop_size": 224,
    "class_ratio": 0.5,
    "seed": 0,
    "use_shuffle": False,
    "transforms": [{
        "name": "apply_transform_to_key",
        "transforms": [
            {"name": "ToTensor"},
            {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        ],
        "key": "input"
    }]
})


# Let us create a ResNet 50 model now.

# In[ ]:


from classy_vision.models import ResNet

model = ResNet.from_config({
    "num_blocks": [3, 4, 6, 3],
    "small_input": False,
    "zero_init_bn_residuals": True
})


# Now, we will create a head with 1000 classes.

# In[ ]:


from classy_vision.heads import FullyConnectedHead

head = FullyConnectedHead(unique_id="default_head", num_classes=1000, in_plane=2048)


# Let us attach the head to the final block of the model.
# 
# For ResNet 50, we want to attach to the `3`rd block in the `4`th layer (based on `[3, 4, 6, 3]`). The blocks use 0 indexing, so this maps to `"block3-2"`.

# In[ ]:


model.set_heads({"block3-2": [head]})


# We can use a cross entropy loss from Pytorch.

# In[ ]:


from torch.nn.modules.loss import CrossEntropyLoss

loss = CrossEntropyLoss()


# For the optimizer, we will be using SGD.

# In[ ]:


from classy_vision.optim import build_optimizer


optimizer = build_optimizer({
    "name": "sgd",
    "param_schedulers": {"lr": {"name": "step", "values": [0.1, 0.01]}},
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "num_epochs": num_epochs
})


# We want to track the top-1 and top-5 accuracies of the model.

# In[ ]:


from classy_vision.meters import AccuracyMeter

meters = [AccuracyMeter(topk=[1, 5])]


# Let's create a directory to save the checkpoints.

# In[ ]:


import os
import time

pretrain_checkpoint_dir = f"/tmp/checkpoint_{time.time()}"
os.mkdir(pretrain_checkpoint_dir)


# Add `LossLrMeterLoggingHook` to monitor the loss and `CheckpointHook` to save the checkpoints.

# In[ ]:


from classy_vision.hooks import CheckpointHook, LossLrMeterLoggingHook, ProgressBarHook

hooks = [
    LossLrMeterLoggingHook(),
    CheckpointHook(pretrain_checkpoint_dir, input_args={})
]


# We have all the components ready to setup our pre-training task which trains for 4 epochs.

# In[ ]:


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

# In[ ]:


from classy_vision.trainer import LocalTrainer

trainer = LocalTrainer()


# Now, we can start training!

# In[ ]:


trainer.train(pretrain_task)


# Training is done! Let us now load the saved checkpoint, we will use this later for fine tuning.

# In[ ]:


from classy_vision.generic.util import load_checkpoint

pretrained_checkpoint = load_checkpoint(pretrain_checkpoint_dir)


# ## 2. Fine-tuning the model

# The original model was trained for 1000 classes. Let's fine-tune it for a problem with only 2 classes. To keep things fast we'll run a single epoch:

# In[ ]:


num_epochs = 1


# We can re-use the same synthetic datasets as before.

# Let us again create a ResNet 50 model.

# In[ ]:


from classy_vision.models import ResNet

model = ResNet.from_config({
    "num_blocks": [3, 4, 6, 3],
    "small_input": False,
    "zero_init_bn_residuals": True
})


# For fine tuning, we will create a head with just 2 classes

# In[ ]:


from classy_vision.heads import FullyConnectedHead

head = FullyConnectedHead(unique_id="default_head", num_classes=2, in_plane=2048)


# Let us attach the head to the final block of the model, like before.

# In[ ]:


model.set_heads({"block3-2": [head]})


# For the optimizer, we will be using RMSProp this time.

# In[ ]:


from classy_vision.optim import build_optimizer


optimizer = build_optimizer({
    "name": "rmsprop",
    "param_schedulers": {"lr": {"name": "step", "values": [0.1, 0.01]}},
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "alpha": 0.9,
    "eps": 1e-3,
    "num_epochs": num_epochs
})


# We want to track the top-1 accuracy of the model.

# In[ ]:


from classy_vision.meters import AccuracyMeter

meters = [AccuracyMeter(topk=[1])]


# We will create a new directory to save the checkpoints for our fine tuning run.

# In[ ]:


import os
import time

fine_tuning_checkpoint_dir = f"/tmp/checkpoint_{time.time()}"
os.mkdir(fine_tuning_checkpoint_dir)


# Hooks are also the same as before.

# In[ ]:


from classy_vision.hooks import CheckpointHook, LossLrMeterLoggingHook

hooks = [
    LossLrMeterLoggingHook(),
    CheckpointHook(fine_tuning_checkpoint_dir, input_args={})
]


# Now we can setup our fine tuning task.

# In[ ]:


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

# In[ ]:


fine_tuning_task.set_freeze_trunk(True)


# We want to start training the heads from scratch, so we will be resetting them. This is required in this example since the pre-trained heads are not compatible with the heads in fine tuning (they have different number of classes). Otherwise, this is also optional.

# In[ ]:


fine_tuning_task.set_reset_heads(True)


# We need to give our task the pre-trained checkpoint, which it'll need to start pre-training on.

# In[ ]:


fine_tuning_task.set_pretrained_checkpoint(pretrained_checkpoint)


# Let us fine tune!

# In[ ]:


trainer.train(fine_tuning_task)


# # 3. Conclusion
# 
# In this tutorial, we learned how to load a pre-trained model in Classy Vision and how to fine-tune it for a different task. We did that by using the `FineTuningTask` abstraction, which lets you load the pretrained model weights, attaching a new head to the model and optionally freeze the weights of the original model. 
# 
# To learn more about about fine-tuning, check out our documentation for [FineTuningTask](https://classyvision.ai/api/tasks.html#classy_vision.tasks.FineTuningTask) and [ClassyHead](https://classyvision.ai/api/heads.html)
