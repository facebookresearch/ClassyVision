#!/usr/bin/env python
# coding: utf-8

# # Using torchscript with Classy Vision

# [torchscript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) is commonly used to export PyTorch models from Python to C++. This is useful for productionizing models, when you typically perform inference on a CPU. This tutorial will demonstrate how to export a Classy Vision model using `torchscript`'s tracing mode and how to load a torchscript model.
# 
# ## 1. Build and train the model
# 
# Our [Getting started](https://classyvision.ai/tutorials/getting_started) tutorial covered many ways of training a model, here we'll simply instantiate a ResNeXT model from a config:

# In[ ]:


from classy_vision.models import build_model
import torch

config = {
    "name": "resnext",
    "num_blocks": [3, 4, 23, 3],
    "num_classes": 1000,
    "base_width_and_cardinality": [4, 32],
    "small_input": False,
    "heads": [
        {
        "name": "fully_connected",
        "unique_id": "default_head",
        "num_classes": 1000,
        "fork_block": "block3-2",
        "in_plane": 2048
        }
    ]
}

model = build_model(config)

# ## 2. Export the model
# 
# Now that the model is built/trained, you can export it using `torch.jit.trace`. To check the results, we'll perform inference on the actual model and on the torchscripted model:

# In[ ]:


with torch.no_grad():
    script = torch.jit.trace(model, torch.randn(1, 3, 224, 224, dtype=torch.float))
    input = torch.randn(1, 3, 224, 224, dtype=torch.float)
    origin_outs = model(input)
    script_outs = script(input)

assert torch.allclose(origin_outs, script_outs)

# After verifying the torchscripted model works as expected, you can save it using `torch.jit.save`:

# In[ ]:


torch.jit.save(script, "/tmp/resnext_101.pt")

# ## 3. Loading a model
# 
# Loading a torchscripted model is as simple as calling `torch.jit.load`. If you need to fine-tune or continue training the model, the loaded model can be attached directly to a `ClassificationTask` or `FineTuningTask` in Classy Vision:

# In[ ]:


loaded_model = torch.jit.load("/tmp/resnext_101.pt")
loaded_outs = loaded_model(input)

assert torch.allclose(loaded_outs, origin_outs)

# ## 4. Conclusion
# 
# `torchscript` makes it really easy to transfer models between research and production with PyTorch, and it works seamlessly with Classy Vision. Check out the [torchscript tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) for more information about how to export a model correctly. 
