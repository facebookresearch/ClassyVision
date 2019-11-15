#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# In[ ]:


import torch


# In[ ]:


# TODO(@mannatsingh): use torchhub when the repo is public. The replacement code
# must be run from the top level directory in the meantime.
# classy_interface = torch.hub.load("facebookresearch/ClassyVision", "resnext101_32x8d_wsl")


# In[ ]:


import os
import sys
path = os.path.abspath(os.path.join(os.path.abspath(""), "../.."))
sys.path.append(path)
import hubconf
classy_interface = hubconf.resnext101_32x8d_wsl()


# In[5]:


# Download an example image from the pytorch website
get_ipython().system('wget https://github.com/pytorch/hub/raw/master/dog.jpg -O dog.jpg')
from IPython.display import Image
Image(filename='dog.jpg')


# In[7]:


dataset = classy_interface.create_image_dataset(["./dog.jpg"], split="test")
data_iterator = classy_interface.get_data_iterator(dataset)
input = next(data_iterator)
# set the model to eval mode
classy_interface.eval()
output = classy_interface.predict(input)
# see the prediction for the input
classy_interface.predict(input).argmax().item()


# In[ ]:




