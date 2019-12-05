Transforms
==========

Classy Vision is able to work directly with `torchvision` transforms
<https://pytorch.org/docs/stable/torchvision/transforms.html>, so it ships with
very few built-in transforms. However, during research it's common to
experiment with new transforms. The `ClassyTransform` class allows users to
express their transforms in a common format and define them in a configuration
file.

Like other Classy Vision abstractions, `ClassyTransform` is accompannied by a
`register_transform` decorator and `build_transform` function for integration
with the config system.

.. automodule:: classy_vision.dataset.transforms
    :members:
