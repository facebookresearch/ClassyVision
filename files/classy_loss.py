#!/usr/bin/env python
# coding: utf-8

# # Creating a custom loss

# Loss functions are crucial because they define the objective to optimize for during training. Classy Vision can work directly with loss functions defined in [PyTorch](https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html) without the need for any wrapper classes, but during research it's common to create custom losses with hyperparameters. Using `ClassyLoss` you can expose these hyperparameters via a configuration file.
# 
# This tutorial will demonstrate: 
# 1. How to create a custom loss within Classy Vision; 
# 2. How to integrate your loss with Classy Vision's configuration system;
# 3. How to use a ClassyLoss independently, without other Classy Vision abstractions.
# 
# ## 1. Defining a loss
# 
# Creating a new loss in Classy Vision is as simple as adding a new loss within PyTorch. The loss has to derive from `ClassyLoss` (which inherits from [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#module)), and implement a `forward` method.
# 
# > **Note**: The forward method should take the right arguments depending on the task the loss will be used for. For instance, a `ClassificationTask` passes the `output` and `target` to `forward`.

# In[ ]:


from classy_vision.losses import ClassyLoss

class MyLoss(ClassyLoss):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, output, target):
        return (output - target).pow(2) * self.alpha

# Now we can start using this loss for training. Take a look at our [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial for more details on how to train a model from a Jupyter notebook.

# In[ ]:


from classy_vision.tasks import ClassificationTask

my_loss = MyLoss(alpha=5)
my_task = ClassificationTask().set_loss(my_loss)

# ## 2. Integrate it with the configuration system
# 
# To be able to use the registration mechanism to be able to pick up the loss from a configuration, we need to do two additional things -
# - Implement a `from_config` method
# - Add the `register_loss` decorator to `MyLoss`

# In[ ]:


from classy_vision.losses import ClassyLoss, register_loss

@register_loss("my_loss")
class MyLoss(ClassyLoss):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    @classmethod
    def from_config(cls, config):
        if "alpha" not in config:
            raise ValueError('Need "alpha" in config for MyLoss')
        return cls(alpha=config["alpha"])
        
    def forward(self, output, target):
        return (output - target).pow(2).sum() * self.alpha

# Now we can start using this loss in our configurations.

# In[ ]:


from classy_vision.losses import build_loss
import torch

loss_config = {
    "name": "my_loss",
    "alpha": 5
}
my_loss = build_loss(loss_config)
assert isinstance(my_loss, MyLoss)

# ClassyLoss inherits from torch.nn.Module, so it works as expected
with torch.no_grad():
    y_hat, target = torch.rand((1, 10)), torch.rand((1, 10))
    print(my_loss(y_hat, target))

# Now that your loss is integrated with the configuration system, you can train it using `classy_train.py` as described in the [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial, no further changes are needed! Just make sure the code defining your model is in the `losses` folder of your classy project.

# ## 3. Conclusion
# 
# In this tutorial, we learned how to make your loss compatible with Classy Vision and how to integrate it with the configuration system. Refer to our documentation to learn more about [ClassyLoss](https://classyvision.ai/api/losses.html).
