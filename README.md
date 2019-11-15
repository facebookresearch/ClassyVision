<p align="center"><img width="70%" src="website/static/img/cv-logo.png" /></p>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/ClassyVision/blob/master/LICENSE) [![CircleCI](https://circleci.com/gh/facebookresearch/ClassyVision.svg?style=svg&circle-token=feeafa057f8d3f6c0c15dfd74db8dd596d9684c8)](https://circleci.com/gh/facebookresearch/ClassyVision) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/ClassyVision/blob/master/CONTRIBUTING.md)

--------------------------------------------------------------------------------

[Classy Vision](http://classyvision.ai) is a PyTorch framework for image and video classification.

## Getting started

Make sure you have an up-to-date installation of PyTorch (1.3.1) and torchvision. To install Classy Vision:
```bash
pip install classy_vision
```

To start a new project:

```bash
classy-project my-project
cd my-project; ./classy_train.py --config templates/template_config.json
```
Voila! A few seconds later your first training run should be done. Check out the results in the checkpoint folder:

```bash
TODO TODO TODO TODO TODO TODO TODO TODO
```

## Documentation

Please see X to learn how to get started on Classy Vision and customize your training runs. Full documentation is available [here](http://TODO).

## Join the Classy Vision community
* Website: http://classyvision.ai
* Slack: [invite link](https://join.slack.com/t/classyvision/shared_invite/enQtODMwODA5Mjg3MTI3LWM5NzNlOTZjNWY3ZTE5YTViYmU2NWM1MDBjMWIwZTIwNmIyY2JjOTkyMTVmMTYzMmIwZWRmZjZmYjhhZTBkZGE)

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
Classy Vision is MIT licensed, as found in the LICENSE file.

## Citing Classy Vision
If you use Classy Vision in your work, please use the following BibTeX entry:

```
@article{adcock2019classy,
  title={Classy Vision},
  author={{Adcock}, A. and {Reis}, V. and {Singh}, M. and {Yan}, Z. {van der Maaten} L., and {Zhang}, K. and {Motwani}, S. and {Guerin}, J. and {Goyal}, N.},
  howpublished = {\url{https://github.com/facebookresearch/ClassyVision}},
  year={2019}
}
```
