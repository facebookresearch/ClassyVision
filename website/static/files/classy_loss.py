#!/usr/bin/env python
# coding: utf-8

# ### Creating a Classy Loss

# Creating a new loss in Classy Vision is as simple as adding a new loss within PyTorch. The loss has to derive from `ClassyLoss` (which inherits from [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#module)), and implement a `forward` method.
# 
# **Note**: The forward method should take the right arguments depending on the task the loss will be used for. For instance, a `ClassificationTask` passes the `output` and `target` to `forward`.

# In[5]:


from classy_vision.losses import ClassyLoss

class MyLoss(ClassyLoss):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, output, target):
        return (output - target).pow(2) * self.alpha


# Now we can start using this loss for training.

# In[ ]:


from classy_vision.tasks import ClassificationTask

my_loss = MyLoss(alpha=5)
my_task = ClassificationTask().set_loss(my_loss)


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
        return (output - target).pow(2) * self.alpha


# Now we can start using this loss in our configurations.

# In[ ]:


from classy_vision.losses import build_loss

loss_config = {
    "name": "my_loss",
    "alpha": 5
}
my_loss = build_loss(loss_config)
assert isinstance(my_loss, MyLoss)

