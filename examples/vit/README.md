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
- Users need to download ImageNet 1K and modify the config to point to the correct paths to the train and val sets
- In addition to the paper's training setup, we use label smoothing, mixup and AutoAugment with reasonable defaults
  - Note that the training hyperparameters here are borrowed from the paper, so the results can possibly be improved with further tuning

### Pre-training on ImageNet 1K

| Model | Training configuration | Top-1 Accuracy (%) |
| --- |--- | --- |
| ViT-B/32 | [vit_b32_in.json](vit_b32_in.json) | 73.30 |
| ViT-B/16 | [vit_b16_in.json](vit_b16_in.json) | 78.98 |
| ViT-L/32 | [vit_l32_in.json](vit_l32_in.json) | 73.60 |
| ViT-L/16 | [vit_l16_in.json](vit_l16_in.json) | 76.53 (\*)|

*(\*) training diverged and accuracy went to 0 around phase 87*

### Fine tuning on ImageNet 1K

- The numbers reported in Table 5 of the paper for ImageNet include an additional fine tuning step using a higher resolution of 384 after pre-training
- The `pretrained_checkpoint` in the config needs to point to the location of a checkpoint of a pre-training run

| Model | Training configuration | Top-1 Accuracy (%) | Paper Top-1 Accuracy (%)
| --- |--- | --- | --- |
| ViT-B/32 | [vit_b32_in_ft.json](vit_b32_in_ft.json) | 76.67 | 73.38 |
| ViT-B/16 | [vit_b16_in_ft.json](vit_b16_in_ft.json) | 79.76 | 77.91 |
| ViT-L/32 | [vit_l32_in_ft.json](vit_l32_in_ft.json) | 75.38 | 71.16 |
| ViT-L/16 | [vit_l16_in_ft.json](vit_l16_in_ft.json) | 77.02 (\*)| 76.53 |

*(\*) pre-training diverged and accuracy went to 0 around phase 87*
