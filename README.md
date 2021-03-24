<p align="center"><img width="50%" src="website/static/img/cv-logo.png" /></p>
<p align="center">
 <a href="https://github.com/facebookresearch/ClassyVision/blob/master/LICENSE">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="GitHub license" />
 </a>
 <a href="https://circleci.com/gh/facebookresearch/ClassyVision">
  <img src="https://circleci.com/gh/facebookresearch/ClassyVision.svg?style=shield&circle-token=feeafa057f8d3f6c0c15dfd74db8dd596d9684c8" alt="CircleCI" />
 </a>
 <a href="https://github.com/facebookresearch/ClassyVision/blob/master/CONTRIBUTING.md">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome" />
 </a>
</p>

## What's New:

- March 2021: Added [RegNetZ models](https://arxiv.org/abs/2103.06877)
- November 2020: [Vision Transformers](https://openreview.net/forum?id=YicbFdNTTy) now available, with training [recipes](https://github.com/facebookresearch/ClassyVision/tree/master/examples/vit)!

<details>
 <summary><b>
  2020-11-20: Classy Vision v0.5 Released
 </b></summary>

#### New Features
- Release [Vision Transformers](https://openreview.net/forum?id=YicbFdNTTy) model implementation, with [recipes](https://github.com/facebookresearch/ClassyVision/tree/master/examples/vit)(#646)
- Implemented gradient clipping (#643)
- Implemented gradient accumulation (#644)
- Added support for [AdamW](https://arxiv.org/abs/1711.05101) (#636)
- Added Precise batch norm hook (#592)
- Added support for adaptive pooling in `fully_convolutional_linear_head` (#602)
- Added support for sync batch norm group size (#534)
- Added a CSV Hook to manually inspect model predictions
- Added a ClassyModel tutorial (#485)
- Migrated to [Hydra 1.0](https://github.com/facebookresearch/hydra) (#536)
- Migrated off of [tensorboardX](https://github.com/lanpa/tensorboardX) (#488)


#### Breaking Changes
- `ClassyOptimizer` API improvements
    - added `OptionsView` to retrieve options from the optimizer `param_group`
- Removed `ClassyModel.evaluation_mode` (#521)
- Removed `ImageNetDataset`, now a subset of `ImagePathDataset` (#494)
- Renamed `is_master` to `is_primary` in `distributed_util` (#576)

</details>

<details>
 <summary><b>
  2020-04-29: Classy Vision v0.4 Released
 </b></summary>

#### New Features
- Release [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf) model implementation ([#475](https://github.com/facebookresearch/ClassyVision/pull/475))
- Add support to convert any `PyTorch` model to a `ClassyModel` with the ability to attach heads to it ([#461](https://github.com/facebookresearch/ClassyVision/pull/461))
  - Added a corresponding [tutorial](https://classyvision.ai/tutorials/classy_model) on `ClassyModel` and `ClassyHeads` ([#485](https://github.com/facebookresearch/ClassyVision/pull/485))
- [Squeeze and Excitation](https://arxiv.org/pdf/1709.01507.pdf) support for `ResNe(X)t` and `DenseNet` models ([#426](https://github.com/facebookresearch/ClassyVision/pull/426), [#427](https://github.com/facebookresearch/ClassyVision/pull/427))
- Made `ClassyHook`s registrable ([#401](https://github.com/facebookresearch/ClassyVision/pull/401)) and configurable ([#402](https://github.com/facebookresearch/ClassyVision/pull/402))
- Migrated to [`TorchElastic v0.2.0`](https://pytorch.org/elastic/master/examples.html#classy-vision) ([#464](https://github.com/facebookresearch/ClassyVision/pull/464))
- Add `SyncBatchNorm` support ([#423](https://github.com/facebookresearch/ClassyVision/pull/423))
- Implement [`mixup`](https://arxiv.org/abs/1710.09412) train augmentation ([#469](https://github.com/facebookresearch/ClassyVision/pull/469))
- Support [`LARC`](https://arxiv.org/abs/1708.03888) for SGD optimizer ([#408](https://github.com/facebookresearch/ClassyVision/pull/408))
- Added convenience wrappers for `Iterable` datasets ([#455](https://github.com/facebookresearch/ClassyVision/pull/455))
- `Tensorboard` improvements
  - Plot histograms of model weights to Tensorboard ([#432](https://github.com/facebookresearch/ClassyVision/pull/432))
  - Reduce data logged to tensorboard ([#436](https://github.com/facebookresearch/ClassyVision/pull/436))
- Invalid (`NaN` / `Inf`) loss detection
- Revamped logging ([#478](https://github.com/facebookresearch/ClassyVision/pull/478))
- Add `bn_weight_decay` configuration option for `ResNe(X)t` models
- Support specifying `update_interval` to Parameter Schedulers ([#418](https://github.com/facebookresearch/ClassyVision/pull/418))

#### Breaking changes
- `ClassificationTask` API improvement and `train_step`, `eval_step` simplification
  - Removed `local_variables` from `ClassificationTask` ([#411](https://github.com/facebookresearch/ClassyVision/pull/411), [#412](https://github.com/facebookresearch/ClassyVision/pull/412), [#413](https://github.com/facebookresearch/ClassyVision/pull/413), [#414](https://github.com/facebookresearch/ClassyVision/pull/414), [#416](https://github.com/facebookresearch/ClassyVision/pull/416), [#421](https://github.com/facebookresearch/ClassyVision/pull/421))
  - Move `use_gpu` from `ClassyTrainer` to `ClassificationTask` ([#468](https://github.com/facebookresearch/ClassyVision/pull/468))
  - Move `num_dataloader_workers` out of `ClassyTrainer` ([#477](https://github.com/facebookresearch/ClassyVision/pull/477))
- Rename `lr` to `value` in parameter schedulers ([#417](https://github.com/facebookresearch/ClassyVision/pull/417))
</details>

<details>
 <summary><b>
  2020-03-06: Classy Vision v0.3 Released
 </b></summary>

#### Release notes
 - `checkpoint_folder` renamed to `checkpoint_load_path` ([#379](https://github.com/facebookresearch/ClassyVision/pull/379))
 - head support on `DenseNet` ([#383](https://github.com/facebookresearch/ClassyVision/pull/383))
 - Cleaner abstraction in `ClassyTask`/`ClassyTrainer`: `eval_step`, `on_start`, `on_end`, …
 - Speed metrics in TB ([#385](https://github.com/facebookresearch/ClassyVision/pull/385))
 - `test_phase_period` in `ClassificationTask` ([#395](https://github.com/facebookresearch/ClassyVision/pull/395))
 - support for losses with trainable parameters ([#394](https://github.com/facebookresearch/ClassyVision/pull/394))
 - Added presets for some typical `ResNe(X)t` configurations: [#405](https://github.com/facebookresearch/ClassyVision/pull/405))
 </details>

## About

[Classy Vision](http://classyvision.ai) is a new end-to-end, PyTorch-based framework for large-scale training of state-of-the-art image and video classification models. Previous computer vision (CV) libraries have been focused on providing components for users to build their own frameworks for their research. While this approach offers flexibility for researchers, in production settings it leads to duplicative efforts, and requires users to migrate research between frameworks and to relearn the minutiae of efficient distributed training and data loading. Our PyTorch-based CV framework offers a better solution for training at scale and for deploying to production. It offers several notable advantages:

* Ease of use. The library features a modular, flexible design that allows anyone to train machine learning models on top of PyTorch using very simple abstractions. The system also has out-of-the-box integration with Amazon Web Services (AWS), facilitating research at scale and making it simple to move between research and production.
* High performance. Researchers can use the framework to train Resnet50 on ImageNet in as little as 15 minutes, for example.
* Demonstrated success in training at scale. We’ve used it to replicate the state-of-the-art results from the paper [Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/abs/1805.00932).
* Integration with PyTorch Hub. AI researchers and engineers can download and fine-tune the best publically available ImageNet models with just a few lines of code.
* Elastic training. We have also added experimental integration with [PyTorch Elastic](https://github.com/pytorch/elastic), which allows distributed training jobs to adjust as available resources in the cluster changes. It also makes distributed training robust to transient hardware failures.

Classy Vision is beta software. The project is under active development and our APIs are subject to change in future releases.

## Installation

#### Installation Requirements
Make sure you have an up-to-date installation of PyTorch (1.6), Python (3.6) and torchvision (0.7). If you want to use GPUs, then a CUDA installation (10.1) is also required.

#### Installing the latest stable release
To install Classy Vision via pip:
```bash
pip install classy_vision
```

To install Classy Vision via conda (only works on linux):
```bash
conda install -c conda-forge classy_vision
```

#### Manual install of latest commit on master

Alternatively you can do a manual install.

```bash
git clone https://github.com/facebookresearch/ClassyVision.git
cd ClassyVision
pip install .
```

## Getting started

Classy Vision aims to support a variety of projects to be built and open sourced on top of the core library. We provide utilities for setting up a project in a standard format with some simple generated examples to get started with. To start a new project:

```bash
classy-project my-project
cd my-project
```

We even include a simple, synthetic, training example to show how to use Classy Vision:

```bash
 ./classy_train.py --config configs/template_config.json
 ```

Voila! A few seconds later your first training run using our classification task should be done. Check out the results in the output folder:
```bash
ls output_<timestamp>/checkpoints/
checkpoint.torch model_phase-0_end.torch model_phase-1_end.torch model_phase-2_end.torch model_phase-3_end.torch
```

`checkpoint.torch` is the latest model (in this case, same as `model_phase-3_end.torch`), a checkpoint is saved at the end of each phase.

For more details / tutorials see the documentation section below.

## Documentation

Please see our [tutorials](https://classyvision.ai/tutorials/) to learn how to get started on Classy Vision and customize your training runs. Full documentation is available [here](https://classyvision.ai/api/).

## Join the Classy Vision community
* Website: https://classyvision.ai
* [Stack overflow](https://stackoverflow.com/questions/tagged/classy-vision)
* Slack: [invite link](https://join.slack.com/t/classyvision/shared_invite/enQtODczNTEyOTUyNTY0LTc4YTc3NThhMzhiNGNjZTkzY2RkYjZiNDc1ZDcyZGYxY2Q0M2M5YjAyYjA4OGQ2M2FlNDk4YzBlNWRjOTg3ZTE)

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
Classy Vision is MIT licensed, as found in the LICENSE file.

## Citing Classy Vision
If you use Classy Vision in your work, please use the following BibTeX entry:

```
@article{adcock2019classy,
  title={Classy Vision},
  author={{Adcock}, A. and {Reis}, V. and {Singh}, M. and {Yan}, Z. and {van der Maaten} L., and {Zhang}, K. and {Motwani}, S. and {Guerin}, J. and {Goyal}, N. and {Misra}, I. and {Gustafson}, L. and {Changhan}, C. and {Goyal}, P.},
  howpublished = {\url{https://github.com/facebookresearch/ClassyVision}},
  year={2019}
}
```
