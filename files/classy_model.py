#!/usr/bin/env python
# coding: utf-8

# # Classy Models
# 
# Before reading this, please go over the [Getting Started tutorial](https://classyvision.ai/tutorials/getting_started).
# 
# Working with Classy Vision requires models to be instances of `ClassyModel`. A `ClassyModel` is an instance of `torch.nn.Module`, but packed with a lot of extra features! 
# 
# If your model isn't implemented as a `ClassyModel`, don't fret - you can easily convert it to one in one line.
# 
# In this tutorial, we will cover:
# 1. Using Classy Models
# 1. Getting and setting the state of a model
# 1. Heads: Introduction & Using Classy Heads
# 1. Creating a custom Classy Model
# 1. Converting any PyTorch model to a Classy Model

# ## Using Classy Models
# As `ClassyModel`s are also instances of `nn.Module`, they can be treated as any normal PyTorch model.

# In[1]:


import torch
from classy_vision.models import build_model


model = build_model({"name": "resnet50"})
input = torch.ones(10, 3, 224, 224)  # a batch of 10 images with 3 channels with dimensions of 224 x 224
output = model(input)

# ## Getting and setting the state of a model
# 
# Classy Vision provides the functions `get_classy_state()` and `set_classy_state()` to fetch and save the state of the models. These are considered drop-in replacements for the [`torch.nn.Module.state_dict`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.state_dict) and [`torch.nn.Module.load_state_dict()`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.load_state_dict) functions and work similarly. For more information, refer to the [docs](https://classyvision.ai/api/models.html#classy_vision.models.ClassyModel).

# In[2]:


state = model.get_classy_state()

model.set_classy_state(state)

# ## Heads: Introduction & Using Classy Heads
# 
# A lot of work in Computer Vision utilizes the concept of re-using a trunk model, like a ResNet 50, and using it for various tasks. This is accomplished by attaching different "heads" to the end of the trunk. 
# 
# Some use cases involve re-training a model trained with a certain head by removing the old head and attaching a new one. This is a special case of fine tuning. If you are interested in fine tuning your models, there's a [tutorial for that as well](https://classyvision.ai/tutorials/fine_tuning)! But first, let's understand the basics.
# 
# Normally, attaching heads or changing them requires users to write code and update their model implementations. Classy Vision does all of this work for you - all of this happens under the hood, with no work required by users!
# 
# All you need to do is decide which `ClassyHead` you want to attach to your model and where. We will use a simple fully connected head in our example, and attach it to the output of the `block3-2` module of our model. Note that a head can be attached to any module, as long as the name of the module is unique.

# In[3]:


from classy_vision.heads import FullyConnectedHead


# a resnet 50 model's trunk outputs a tensor of 2048 dimension, which will be the
# in_plane of out head
#
# let's say we want a 100 dimensional output
#
# Tip: you can use build_head() as well to create a head instead of initializing the
# class directly
head = FullyConnectedHead(unique_id="default", num_classes=100, in_plane=2048)

# let's attach this head to our model
model.set_heads({"block3-2": [head]})

output = model(input)
assert output.shape == (10, 100)

# let's change the head one more time
head = FullyConnectedHead(unique_id="default", num_classes=10, in_plane=2048)

model.set_heads({"block3-2": [head]})

output = model(input)
assert output.shape == (10, 10)

# Classy Vision supports attaching multiple heads to one or more blocks as well, but that is an advanced concept which this tutorial does not cover. For inquisitive users, here is an example -

# In[4]:


head_1_1 = FullyConnectedHead(unique_id="1_1", num_classes=10, in_plane=1024)
head_1_2 = FullyConnectedHead(unique_id="1_2", num_classes=20, in_plane=1024)
head_2 = FullyConnectedHead(unique_id="2", num_classes=100, in_plane=2048)

# we can attach these heads to our model at different blocks
model.set_heads({"block2-2": [head_1_1, head_1_2], "block3-2": [head_2]})

output = model(input)

# ## Creating a custom Classy Model
# 
# This section will demonstrate: (1) how to create a custom model within Classy Vision; (2) how to integrate your model with Classy Vision's configuration system; (3) how to use the model for training and inference;
# 
# ### 1. Defining a model
# 
# Creating a new model in Classy Vision is the simple as creating one within PyTorch. The model needs to derive from `ClassyModel` and implement a `forward` method to perform inference. `ClassyModel` inherits from [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#module), so it works exactly as you would expect.

# In[5]:


import torch.nn as nn

from classy_vision.models import ClassyModel


class MyModel(ClassyModel):
    def __init__(self, num_classes):
        super().__init__()
        
        # Average all the pixels, generate one output per class
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_channels = 3
        self.fc = nn.Linear(num_channels, num_classes)
        
    def forward(self, x):
        # perform average pooling
        out = self.avgpool(x)

        # reshape the output and apply the fc layer
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Now we can start using this model for training. Take a look at our [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial for more details on how to train a model from a Jupyter notebook.

# In[6]:


from classy_vision.tasks import ClassificationTask

my_model = MyModel(num_classes=1000)
my_task = ClassificationTask().set_model(my_model)

# ### 2. Integrating it with the configuration system
# 
# Classy Vision is also able to read a configuration file and instantiate the model. This is useful to keep your experiments organized and reproducible. For that, you have to:
# 
# - Implement a `from_config` method
# - Add the `register_model` decorator to `MyModel`

# In[7]:


import torch.nn as nn

from classy_vision.models import ClassyModel, register_model


@register_model("my_model")
class MyModel(ClassyModel):
    def __init__(self, num_classes):
        super().__init__()
        
        # Average all the pixels, generate one output per class
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_channels = 3
        self.fc = nn.Linear(num_channels, num_classes)

    @classmethod
    def from_config(cls, config):
        # This method takes a configuration dictionary 
        # and returns an instance of the class. In this case, 
        # we'll let the number of classes be configurable.
        return cls(num_classes=config["num_classes"])
        
    def forward(self, x):
        # perform average pooling
        out = self.avgpool(x)

        # reshape the output and apply the fc layer
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Now we can start using this model in our configurations. The argument passed to `register_model` is used to identify the model class in the configuration:

# In[8]:


from classy_vision.models import build_model
import torch

model_config = {
    "name": "my_model",
    "num_classes": 3
}
my_model = build_model(model_config)
assert isinstance(my_model, MyModel)

# my_model inherits from torch.nn.Module, so inference works as usual:
x = torch.rand((1, 3, 200, 200))
with torch.no_grad():
    print(my_model(x))

# Now that your model is integrated with the configuration system, you can train it using `classy_train.py` as described in the [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial, no further changes are needed! Just make sure the code defining your model is in the `models` folder of your classy project.

# ## Converting any PyTorch model to a Classy Model
# 
# Any model can be converted to a Classy Model with a simple function call - `ClassyModel.from_model()`

# In[9]:


from torchvision.models import resnet18
from classy_vision.models import ClassyModel


model = resnet18()
classy_model = ClassyModel.from_model(model)
output = classy_model(input)
assert output.shape == (10, 1000)

# In fact, as soon as a model becomes a Classy Model, it gains all its abilities as well, including the ability to attach heads! Let us inspect the original model to see the modules it comprises.

# In[10]:


model

# It seems that the final trunk layer of this model is called `layer4`. Let's try to attach heads here.

# In[11]:


# the output of layer4 is 512 dimensional
head = FullyConnectedHead(unique_id="default", num_classes=10, in_plane=512)

classy_model.set_heads({"layer4": [head]})

output = classy_model(input)
assert output.shape == (10, 10)  # it works!

# You might be wondering how to figure out the `in_plane` for any module. A simple trick is to try attaching any head and noticing the `Exception` if there is a size mismatch!

# In[12]:


try:
    head = FullyConnectedHead(unique_id="default", num_classes=10, in_plane=1234)

    classy_model.set_heads({"layer4": [head]})

    output = classy_model(input)

except Exception as e:
    print(e)

# The error tells us that the size should be 512.

# ## Conclusion
# 
# In this tutorial, we covered how to use Classy Models, how to get and set their state, and how to create our own models & integrating them with the configuration system. We also got familiarized with the concept of heads and how they work with Classy Vision. Lastly, we learned how we can easily convert any PyTorch models to Classy Models and unlock all the features they provide.
# 
# For more information, refer to our [API docs](https://classyvision.ai/api/).
