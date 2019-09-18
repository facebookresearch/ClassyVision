#!/usr/bin/env python3

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules

from .classy_criterion import ClassyCriterion


FILE_ROOT = Path(__file__).parent


CRITERION_REGISTRY = {}
CRITERION_CLASS_NAMES = set()


def build_criterion(config):
    return CRITERION_REGISTRY[config["name"]](config)


def register_criterion(name):
    """Decorator to register a new criterion."""

    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError("Cannot register duplicate optimizer ({})".format(name))
        if not issubclass(cls, ClassyCriterion):
            raise ValueError(
                "Criterion ({}: {}) must extend ClassyCriterion".format(
                    name, cls.__name__
                )
            )
        CRITERION_REGISTRY[name] = cls
        CRITERION_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_criterion_cls


# automatically import any Python files in the criterion/ directory
import_all_modules(FILE_ROOT, "classy_vision.criterions")
