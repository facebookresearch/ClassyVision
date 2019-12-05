#!/usr/bin/env python
# coding: utf-8

# # Distributed training on AWS

# In this tutorial we will learn: 
# 1. How to start a cluster on AWS for use with Classy Vision; 
# 2. How to start a new project on the cluster; 
# 3. How to launch training jobs on the cluster;
# 
# ## 1. Setup
# 
# Make sure you have Classy Vision installed, as described in our [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial. 
# 
# For this tutorial we will also need the Classy Vision sources, you can clone it with this command (on your terminal):

# In[ ]:


! git clone https://github.com/facebookresearch/ClassyVision.git

# In this tutorial we'll use [Ray](https://github.com/ray-project/ray) to manage the AWS resources. Install Ray and all its required dependencies with:

# In[ ]:


% cd ./ClassyVision/examples/ray
! pip install -r requirements.txt

# You should also set up your AWS CLI and credentials as described [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration). To make sure everything is working, run on your terminal:

# In[ ]:


! aws ec2 describe-instances

# That should print a JSON file with all your current AWS instances (or empty if you don't have any). 

# ## 2. Cluster setup

# We have a sample cluster configuration file stored in the Classy Vision repository, under `./examples/ray/cluster_config.yml`. Let's verify that Ray can start the cluster appropriately:

# In[ ]:


! ray up cluster_config.yml -y

# That will take about 10 minutes, and at the end you should see a message explaining how to connect to the cluster. Assuming everything worked successfully, now tear down the cluster:

# In[ ]:


! ray down cluster_config.yml -y

# We will now set up an EFS volume to store our code and datasets. Follow [this tutorial](https://aws.amazon.com/getting-started/tutorials/create-network-file-system/) to setup the EFS volume in your AWS account. 
# 
# When you're done with that tutorial, go back to the EFS section in the AWS console, find your filesystem there and click `Manage file system access`. Add the `ray-autoscaler-default` security group to the list of security groups allowed to use your EFS volume. That security group should have been created by the `ray up` command we ran earlier.
# 
# You should now have an identifier for your EFS volume. Open `cluster_config.yml` in your favorite text editor and replace `{{FileSystemId}}` with your own EFS id. We are now ready to launch our cluster again:

# In[ ]:


! ray up cluster_config.yml

# ## 3. Create a project
# 
# When it's done, let's attach to the head node of the cluster:

# In[ ]:


! ray attach cluster_config.yml

# That will give you an SSH session into the head node, which coordinates all the worker nodes in Ray. In our example configuration file, the head node is a CPU-only machine, and the workers all have GPUs.
# 
# Both the head node and the worker nodes will have the same EFS volume mounted, so we'll use that to send code from the head to the workers. The following commands are meant to run on the head node (e.g. in the terminal prompt you got from `ray attach`). Let's start a project in the EFS folder:

# In[ ]:


$ cd efs
$ classy-project my_project

# ## 4. Start training
# 
# Classy Vision comes with a launcher analogous to `torch.distributed.launch`, but that launches jobs on multiple machines using Ray. To use it, simply run:

# In[ ]:


$ python -m classy_vision.distributed.launch_ray --nnodes=2 --use_env ~/efs/my_project/classy_train.py --config ~/efs/my_project/configs/template_config.json --distributed_backend ddp

# Your first time running this you might see logs like `Not enough GPUs available`. That's normal, and it's because the worker nodes are still being set up. The `ray up` command should have printed a command line you can use to follow their progress. But there's no need to do anything, the launcher will wait until the workers are available and execute the command automatically.
# 
# That's it! When that command is done it should print the folder where the checkpoints are.
# 
# > Note that we specified the full absolute path for the config in the argument list. That's because the `classy_train.py` command is running on a remote machine and we are relying on the fact that the EFS folder is mounted at exactly the same location on the head and worker nodes. Keep that in mind if you modify this setup.
# 
# > Remember to tear down the cluster with `ray down cluster_config.yml` when you're done. You will be billed as long as the machines are up, even when not using them.

# ## 5. Troubleshooting
# 
# If you hit an error during this tutorial, here are a few things that might help to debug what is going on:
# 
# ### Make sure all workers have initialized properly
# 
# When the `ray up` command finishes, it prints a command line to tail the logs. It should look like:
# ```bash
# ray exec cluster_config.yml 'tail -n 100 -f /tmp/ray/session_*/logs/monitor*'
# ```
# 
# Run that command and look for any errors. When the workers are done initializing, you should see `-- StandardAutoscaler: 2/2 target nodes (0 pending)` printed repeatedly on the logs.
# 
# ### Make sure EFS volumes are mounted on all machines
# 
# Sometimes the EFS package fails to install on workers. To verify EFS is working, get the worker node IPs with `ray get-worker-ips cluster_config.yml`, then ssh on them with:
# ```bash
# ssh -i ~/.ssh/ray-autoscaler_us-west-2.pem ubuntu@<WORKER-IP>
# ```
# 
# Once in a worker machine, run `df -h` to list all the current mounts. Verify `/home/ubuntu/efs` is on that list. If it's not, look for the EFS setup commands on the `cluster_config.yml` file and run them yourself. That should clarify what the issue is. If you didn't setup the EFS security groups correctly (as described in step 2), the `mount` command will hang for a few minutes then fail.
# 
# ## 6. Conclusion
# 
# In this tutorial we covered how to start using Classy Vision on AWS using Ray. For more information about Ray, check out their [repository](https://github.com/ray-project/ray). The next tutorials ([[1]](https://classyvision.ai/tutorials/classy_model), [[2]](https://classyvision.ai/tutorials/classy_loss), [[3]](https://classyvision.ai/tutorials/classy_dataset)) will demonstrate how to customize the project created by the `classy-project` utility for your own needs. To learn more about how to train models in Classy Vision and how to use Tensorboard to visualize training progress, check out our [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial.
