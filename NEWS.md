# News

## 2020-04-29: Classy Vision v0.4 Released
### New Features
- Release [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf) model implementation ([#475](https://github.com/facebookresearch/ClassyVision/pull/475))
- Add support to convert any `PyTorch` model to a `ClassyModel` with the ability to attach heads to it ([[#461](https://github.com/facebookresearch/ClassyVision/pull/461)](https://github.com/facebookresearch/ClassyVision/pull/461))
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

### Breaking changes
- `ClassificationTask` API improvement and `train_step`, `eval_step` simplification
  - Removed `local_variables` from `ClassificationTask` ([#411](https://github.com/facebookresearch/ClassyVision/pull/411), [#412](https://github.com/facebookresearch/ClassyVision/pull/412), [#413](https://github.com/facebookresearch/ClassyVision/pull/413), [#414](https://github.com/facebookresearch/ClassyVision/pull/414), [#416](https://github.com/facebookresearch/ClassyVision/pull/416), [#421](https://github.com/facebookresearch/ClassyVision/pull/421))
  - Move `use_gpu` from `ClassyTrainer` to `ClassificationTask` ([#468](https://github.com/facebookresearch/ClassyVision/pull/468))
  - Move `num_dataloader_workers` out of `ClassyTrainer` ([#477](https://github.com/facebookresearch/ClassyVision/pull/477))
- Rename `lr` to `value` in parameter schedulers ([#417](https://github.com/facebookresearch/ClassyVision/pull/417))

## 2020-03-06: Classy Vision v0.3 Released
### Release notes
 - checkpoint_folder renamed to checkpoint_load_path ([#379](https://github.com/facebookresearch/ClassyVision/pull/379))
 - head support on DenseNet ([#383](https://github.com/facebookresearch/ClassyVision/pull/383))
 - Cleaner abstraction in ClassyTask/ClassyTrainer: eval_step, on_start, on_end, …
 - Speed metrics in TB ([#385](https://github.com/facebookresearch/ClassyVision/pull/385))
 - test_phase_period in ClassificationTask ([#395](https://github.com/facebookresearch/ClassyVision/pull/395))
 - support for losses with trainable parameters ([#394](https://github.com/facebookresearch/ClassyVision/pull/394))
 - Added presets for some typical resNe(X)t configurations: [#405](https://github.com/facebookresearch/ClassyVision/pull/405))
 
## 2020-01-24: Classy Vision v0.2 Released
### New features
 - Adam optimizer ([#301](https://github.com/facebookresearch/ClassyVision/pull/301))
 - R(2+1)d units ([#322](https://github.com/facebookresearch/ClassyVision/pull/322))
 - Mixed precision training ([#338](https://github.com/facebookresearch/ClassyVision/pull/338))
 - One-hot targets in meters ([#349](https://github.com/facebookresearch/ClassyVision/pull/349))

This release has been tested on the latest PyTorch (1.4) and torchvision (0.5) releases. It also includes bug fixes and other smaller features.

## 2019-12-05: Classy Vision v0.1 Released
- [A new framework for large-scale training of state-of-the-art visual classification models](https://ai.facebook.com/blog/a-new-framework-for-large-scale-training-of-state-of-the-art-visual-classification-models/)
- [NeurIPS 2019 Expo Workshop Presentation](https://research.fb.com/wp-content/uploads/2019/12/3.-Classy-Vision.key)
