# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

*Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv preprint arXiv:2010.11929 (2020).* 

https://arxiv.org/abs/2010.11929

## Introduction

This paper takes transformer based models that have been extremely successful in NLP (e.g. GPT-3) and successfully makes them work for Computer Vision by applying attention on image patches.

## Training Recipes

### Pre-training on ImageNet

|Model        |Training configuration|Top-1 Accuracy|
|---          |---                   |---           |
|ViT-B/32     |                      |              |

### Fine tuning on ImageNet

The numbers reported in Table 5 of the paper for ImageNet include an additional fine tuning step with a higher resolution after pre-training.

|Model        |Training configuration|Top-1 Accuracy|
|---          |---                   |---           |
|ViT-B/32     |                      |              |
