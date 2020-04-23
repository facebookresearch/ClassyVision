#!/usr/bin/env python
# coding: utf-8

# # Elastic training with Classy Vision
# 
# This tutorial will demonstrate how to launch an training job on Amazon Web Services ([AWS](https://aws.amazon.com/)) using [PyTorch Elastic](https://github.com/pytorch/elastic) and Classy Vision.
# 
# ## Prerequisites
# 
# 1. Familiarity with basic AWS (EC2, Auto Scaling Groups, S3, EFS).
# 2. (suggested) install and setup [`awscli`](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html).
# 3. Basic knowledge of containers (we use Docker in our examples).
# 
# ## 1. Setup

# Download the PyTorch Elastic repository and install it. Run in your terminal:

# In[ ]:


get_ipython().system(' git clone https://github.com/pytorch/elastic.git')
get_ipython().system(' pip install torchelastic')


# Install the required dependencies for AWS:

# In[ ]:


get_ipython().run_line_magic('cd', 'elastic/aws')
get_ipython().system(' pip install -r requirements.txt')


# Make sure you are familiar with the following AWS resources:
# 
#   1. EC2 [instance profile](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-ec2_instance-profiles.html)
#   2. EC2 [key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)
#   3. [Subnet(s)](https://docs.aws.amazon.com/vpc/latest/userguide/default-vpc.html#create-default-subnet)
#   4. [Security group](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html#DefaultSecurityGroup)
#   5. EFS volume
#   6. S3 bucket
#   
# [Install](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
#  the AWS Session Manager plugin.

# ## 2. Create the cluster
# 
# `petctl` is a commandline tool that helps run distributed jobs written with torchelastic on EC2 instances. It's available in the `aws` directory of the `torchelastic` repo. To get started, run this on your terminal:
# 
# ```bash
# python3 petctl.py setup
# ```
# 
# This will bootstrap all the AWS resources required to run a torchelastic
# job. For details take a look at the CloudFormation [template](cfn/setup.yml) .
# 
# Use `--s3_bucket` and `--efs_id` to use an existing S3 bucket and EFS 
# file system. Otherwise an S3 bucket and EFS volume will be created.
# 
# > **IMPORTANT** when specifying `--efs_id` you MUST ensure that NO mount targets
# exist on the EFS file system. torchelastic's cfn stack will attempt to create
# mount targets for the subnets it creates and WILL FAIL if the file system already
# has mount targets on a different VPC. For more information refer to 
# the [EFS docs](https://docs.aws.amazon.com/efs/latest/ug/accessing-fs.html). 
# 
# **TIP:** If the stack creation fails, log into the CloudFormation console, inspect
# the failure reason, address the failure, then manually delete the stack and re-run
# `petctl configure`.
# 
# If you are familiar with AWS or already have the resources specified in the 
# **Requirements** section above, then you can follow the [Manual Setup](https://github.com/pytorch/elastic/blob/master/aws/README.md) instructions
# in the `torchelastic` repository. Simply copy the sample specs file and fill
# in the template, then run `python petctl.py configure`. 

# ## 3. Create your Classy Vision project
# 
# If you already have a Classy Vision project to use with `torchelastic`, great! You only need to modify `classy_train.py` to use an `ElasticTrainer` instead of a `DistributedTrainer`. See our [getting started](https://classyvision.ai/tutorials/getting_started) tutorial for more details about `classy_train.py`.
# 
# To make things easier, we provided an example of how to use `ElasticTrainer`: it's under `./examples/classy_vision/main.py` in the `torchelastic` repo. You can start by copying that file and use it to replace `classy_train.py`.

# 
# ## 4. Start training
# 
# Normally you would run the training script directly to start training. For elastic training, we'll use `petctl` to launch it. Here's how you launch our example script in your terminal:
# 
# ``` bash
# python3 aws/petctl.py run_job --size 2 --min_size 2 --max_size 2 --name ${USER}-job examples/classy_vision/main.py -- --config_file classy-vision://configs/resnet50_synthetic_image_classy_config.json --num_workers 0```
# 

# In the example above, the named arguments, such as, `--size` , `--min_size`, and
# `--max_size` are parameters to the `run_job` sub-command of `petctl`. In the example
# above, we created an **elastic** job where the initial worker `--size=2`, we are
# allowed to scale down to `--min_size` and up to `--max_size`. This is used by
# torchelastic's rendezvous algorithm to determine how many nodes to admit on each
# re-rendezvous before considering the group *final* and start the `train_step`.
# 
# Because all the size parameters are the same in this case, that means we are disabling elasticity. You might want to do that for reproducibility reasons, for instance. Training this way still provides benefits, since `torchelastic`  increases robustness: when a node fails, we can start a new node and resume training from the last minibatch, without reverting back to the previous checkpoint. 
# 
# The other positional arguments have the form:
# 
# ```
# [local script] -- [script args ...]
#   -- or -- 
# [local directory] -- [script] [script args...]
# ```
# 
# If the first positional argument is a path to a script file, then the script
# is uploaded to S3 and the script arguments specified after the `--` delimiter
# are passed through to the script.
# 
# If the first positional argument is a directory, then a tarball of the directory
# is created and uploaded to S3 and is extracted on the worker-side. In this case
# the first argument after the `--` delimiter is the path to the script **relative** to the
# specified directory and the rest of the arguments after the delimiter is passed 
# to the script.
# 
# 
# In our example we specified
# ```
# petctl.py run_job [...] classy_vision/main.py --config_file [...]
# ```
# 
# We could have decided to specify the directory instead
# ```
# petctl.py run_job [...] classy_vision -- main.py --config_file [...]
# ```
# 
# **TIP 1:** Besides a local script or directory you can run with scripts or `tar` files
# that have already been uploaded to S3 or directly point it to a file or directory
# on the container.
# ``` bash
# python3 petctl.py run_job [...] s3://my-bucket/my_script.py
# python3 petctl.py run_job [...] s3://my-bucket/my_dir.tar.gz -- my_script.py
# 
# # or
# python3 petctl.py run_job [...] docker:///abs/path/in/container/dir -- my_script.py
# python3 petctl.py run_job [...] docker://rel/path/in/container/dir/my_script.py
# ```
# 
# **TIP 2:** To iterate quickly, simply make changes to your local script and
# upload the script to S3 using
# ```bash 
# python3 petctl.py upload examples/imagenet/main.py s3://<bucket>/<prefix>/<job_name> 
# ```
# 
# **TIP 3:** Use the EFS volume attached on `/mnt/efs/fs1` on all the workers to 
# save input data, checkpoints and job output.
# 
# Once the `run_job` command returns log into the EC2 console, you will see two
# Auto Scaling Groups
# 1. etcd server 
# 2. workers
# 
# ## 5. Inspect the logs
# Log into the AWS CloudWatch Logs console. You should see a log group called
# `torchelastic/$USER`. Under it there will be a log stream per instance with the 
# name `$job_name/$instance_id` (e.g. `my_job/i0b938EXAMPLE`).
# 
# #### Troubleshooting
# To SSH onto the worker nodes to debug/inspect the worker process use AWS 
# Session Manager instead of the ec2 key pair. [Install](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
#  the Session Manager plugin and run
# 
# ``` bash
# # get the instance ids of the workers
# python3 petctl.py list_hosts <job_name>
# 
# # ssh onto one of the workers
# awscli ssm start-session --target <instance_id>
#  -- example --
# awscli ssm start-session --target i-00b00EXAMPLE
# ```
# 
# Once SSH'ed, the workers run in a docker container managed by `systemd`.
# You can take a look at their console outputs by running
# 
# ``` bash
# # see the status of the worker
# sudo systemctl status torchelastic_worker
# # get the container id
# sudo docker ps
# # tail the container logs
# sudo docker logs -f <container id>
# ```
# 
# You can also manually stop and start the workers by running
# ``` bash
# sudo systemctl stop torchelastic_worker
# sudo systemctl start torchelastic_worker
# ```

# > **EXERCISE:** Open up two terminals and SSH onto each worker. Tail the docker logs
# on each worker. Now stop worker 1 and observe the worker 2 re-rendezvous and
# since `--min_size=1` it continues training by itself. Now restart worker 1 and
# observe that worker 2 notices that worker 1 is waiting to join and re-rendezvous,
# the `state` object in worker 2 is `sync()`'ed to worker 1 and both resume training
# without loss of progress.

# > **Note**: by design, `petctl` tries to use the least number of AWS services. This
# was done intentionally to allow non-AWS users to easily transfer the functionality
# to their environment. Hence it currently does not have the functionality to query
# status of the job or to terminate the ASG when the job is done (there is nothing
# that is monitoring the job!). In practice consider using EKS, Batch, or SageMaker.

# ## 6. Stop training
# To stop the job and tear down the resources, use the `kill_job` command:
# 
# ``` bash
# python3 petctl.py kill_job ${USER}-job
# ```
# 
# You'll notice that the two ASGs created with the `run_job` command are deleted.
