#!/usr/bin/env python
# coding: utf-8

# # Elastic training with Classy Vision
# 
# This tutorial will demonstrate how to use [PyTorch Elastic](https://github.com/pytorch/elastic) with Classy Vision.
# 
# ## Prerequisites
# 
# 1. (recommended) cloud provider instance with GPUs;
# 2. [Docker](https://docs.docker.com/get-docker/)
# 3. [NVidia container toolkit](https://github.com/NVIDIA/nvidia-docker)
# 
# ## 1. Setup

# Download the PyTorch Elastic repository and install it. Run in your terminal:

# In[ ]:


get_ipython().system(' git clone https://github.com/pytorch/elastic.git')
get_ipython().system(' pip install torchelastic')


# Download and install Classy Vision:

# In[ ]:


get_ipython().system(' git clone https://github.com/facebookresearch/ClassyVision.git')
get_ipython().system(' pip install classy_vision')


# If needed, install Docker:

# In[ ]:


get_ipython().system(' sudo apt install docker-compose')


# To run torchelastic manually you'll also need etcd:

# In[ ]:


get_ipython().system(' sudo apt install etcd-server')


# Set this environment variable to your current `torchelastic` version. This tutorial only works for version >= 0.2.0:

# In[ ]:


get_ipython().system(' export VERSION=<torchelastic version>')


# ## 1. Single node, multi-GPU training
# 
# The easiest way to get started is to use our example docker image. Run the following in your shell:
# 
# ```
#   export NUM_CUDA_DEVICES=2
# ```

# In[ ]:


$ docker run --shm-size=2g --gpus=all torchelastic/examples:$VERSION
           --standalone
           --nnodes=1
           --nproc_per_node=$NUM_CUDA_DEVICES
           workspace(/classy_vision/classy_train.py)
           --device=gpu
           --config_file /workspace/classy_vision/configs/template_config.json


# If you don't have GPUs available, simply drop the `--gpus=all` flag. This will download and launch our example Docker container and start training on the current machine using torchelastic and Classy Vision. This is fine as a sanity check, but elasticity is really intended to help with training on multiple nodes. The next section will walk you through that.

# ## 2. Launching torchelastic manually

# Now let's replicate what the Docker example in the previous section did, to see how things work under the hood. torchelastic provides a drop-in replacement for `torch.distributed.launch` and that's compatible with Classy Vision's `classy_train.py`. The main difference is that torchelastic requires launching an `etcd` server so that the workers know how to communicate with each other. In your shell, run this:
#     

# In[ ]:


get_ipython().system(' classy-project my-project')


# In[ ]:


get_ipython().run_line_magic('cd', 'my-project')


# Launch the etcd server:

# In[ ]:


get_ipython().system(' etcd --enable-v2 --listen-client-urls http://0.0.0.0:2379,http://127.0.0.1:4001 --advertise-client-urls http://127.0.0.1:2379')


# This might fail if you alread have an etcd server running. torchelastic requires etcd v2 in order to work properly, so make sure to kill any etcd instances that you have running already.

# Start training:

# In[ ]:


get_ipython().system(' python -m torchelastic.distributed.launch --nproc_per_node=$NUM_CUDA_DEVICES --rdzv_endpoint 127.0.0.1:2379     ./classy_train.py --config configs/template_config.json --distributed_backend ddp')


# That's it! The training script should start running with torchelastic enabled.
# 
# Take a look at this [link](http://pytorch.org/elastic/0.2.0/train_script.html) for the full documentation on how `torchelastic.distributed.launch` works.

# ## 3. Multi-container

# `torchelastic` is meant to help with distributed training on multiple machines. In this part, we will simulate a multiple machine setup by launching multiple containers in the same host. Set this environment variable for the location of your ClassyVision repository:
# 
# ```
# export CLASSY_VISION_HOME=~/ClassyVision
# ```
# 
# In your shell, run:

# In[ ]:


cd $CLASSY_VISION_HOME/examples/elastic
classy-project my_project


# This will setup a Classy Vision project within the examples folder, which our containers will use as the training script. Now launch the containers:

# In[ ]:


docker-compose up


# That's it! This will launch two containers: one running the etcd server, and another doing training. You should see the output from both the etcd server and from the training script in your terminal.

# ## 4. Conclusion
# 
# In this tutorial, we covered how to launch the torch elastic on a single machine and how to launch torchelastic jobs in multiple containers. For training on multiple nodes, check out this [tutorial](http://pytorch.org/elastic/0.2.0/examples.html#multi-node) on the Pytorch Elastic documentation.
