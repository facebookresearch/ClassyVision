#!/usr/bin/env python
# coding: utf-8

# # Creating a custom model

# This tutorial will demonstrate: (1) how to create a custom model within Classy Vision; (2) how to integrate your model with Classy Vision's configuration system; (3) how to use the model for training and inference;
# 
# ## 1. Defining a model
# 
# Creating a new model in Classy Vision is the simple as creating one within PyTorch. The model needs to derive from `ClassyModel` and implement a `forward` method to perform inference. `ClassyModel` inherits from [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#module), so it works exactly as you would expect.

# In[ ]:


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

# In[ ]:


from classy_vision.tasks import ClassificationTask

my_model = MyModel(num_classes=1000)
my_task = ClassificationTask().set_model(my_model)

# ## 2. Integrate it with the configuration system
# 
# Classy Vision is also able to read a configuration file and instantiate the model. This is useful to keep your experiments organized and reproducible. For that, you have to:
# 
# - Implement a `from_config` method
# - Add the `register_model` decorator to `MyModel`

# In[ ]:


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

# In[ ]:


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

# ## 3. Conclusion
# 
# In this tutorial, we learned how to make your model compatible with Classy Vision and how to integrate it with the configuration system. Refer to our documentation to learn more about [ClassyModel](https://classyvision.ai/api/models.html).
