#!/usr/bin/env python3

MODEL_STEM_REGISTRY = {}
MODEL_STEM_CLASS_NAMES = set()


def register_model_stem(name):
    """
    New model stems can be added with the `register_model_stem` function decorator.
    For example::

        @register_model_stem('resnext3d_stem')
        class ResNeXt3DStem(nn.Module):
            (...)

    Args:
        name (str): the name of the model stem
    """

    def register_model_stem_cls(cls):
        if name in MODEL_STEM_REGISTRY:
            raise ValueError("Cannot register duplicate model stem ({})".format(name))

        if cls.__name__ in MODEL_STEM_CLASS_NAMES:
            raise ValueError(
                "Cannot register model with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        MODEL_STEM_REGISTRY[name] = cls
        MODEL_STEM_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_model_stem_cls


def build_model_stem(name, config):
    assert name in MODEL_STEM_REGISTRY, "unknown model stem (%s)" % name
    return MODEL_STEM_REGISTRY[name](config)
