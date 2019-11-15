# sphinx API reference

This file describes the sphinx setup for auto-generating the API reference.


## Installation

**Requirements**:
- sphinx >= 2.0
- sphinx_autodoc_typehints

You can install these via `pip install sphinx sphinx_autodoc_typehints`.


## Building

From the `ClassyVision/sphinx` directory, run `make html`.

Generated HTML output can be found in the `captum/sphinx/build` directory. The main index page is: `ClassyVision/sphinx/build/html/index.html`


## Structure

`source/index.rst` contains the main index. The API reference for each module lives in its own file, e.g. `dataset.rst` for the `dataset.models` module.
