# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

*Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv preprint arXiv:2010.11929 (2020).* 

https://arxiv.org/abs/2010.11929

## Introduction

This paper takes transformer based models that have been extremely successful in NLP (e.g. GPT-3) and successfully makes them work for Computer Vision by applying attention on image patches.

## Training Recipes

- These recipes were used to train models using 16GB V100 GPUs using mixed precision training
- Based on the type of GPUs available, the `batchsize_per_replica` in any config can be adjusted and mixed precision training can be disabled
- We use gradient accumulation in all our training runs with a pre-defined global batch size (`simulated_global_batchsize`)
  - This means these configs can be used with any number of GPUs, as long as `simulated_global_batchsize` is divisible by `batchsize_per_replica * num_gpus`

### Pre-training on ImageNet

| Model | Training configuration | Top-1 Accuracy (%) |
| --- |--- | --- |
| ViT-B/32 | [vit_b32_in.json](vit_b32_in.json) | 73.30 |
| ViT-B/16 | [vit_b16_in.json](vit_b16_in.json) | 78.98 |


### Fine tuning on ImageNet

The numbers reported in Table 5 of the paper for ImageNet include an additional fine tuning step using a higher resolution of 384 after pre-training.

| Model | Training configuration | Top-1 Accuracy (%) |
| --- |--- | --- |
| ViT-B/32 | [vit_b32_in_ft.json](vit_b32_in_ft.json) | 76.67 |
| ViT-B/16 | [vit_b16_in_ft.json](vit_b16_in_ft.json) | 79.76 |
