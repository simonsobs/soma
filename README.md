soma
====

[![CI](https://github.com/msyriac/soma/actions/workflows/ci.yml/badge.svg)](https://github.com/msyriac/soma/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/soma/badge/?version=latest)](https://soma.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/pysoma.svg)](https://pypi.org/project/pysoma/)

[`soma`](https://en.wikipedia.org/wiki/Soma_(Brave_New_World)) (Simons Observatory Map-based Analysis) is a set of companion tools for use with the [`pixell`](https://github.com/simonsobs/pixell) library.

Installation
------------

To install from a local checkout, including the development dependencies:

```bash
git clone https://github.com/msyriac/soma.git
cd soma
[uv] pip install -e ".[dev]"
```

A PyPI instance will be deployed once development is mature.

Usage
-----

```python
import soma

print(soma.__version__)
```

Some non-trivial features of soma include:
- Going from planet maps (or an optics model) to beam azimuthal modes and T->P leakage
  — see the [beams quickstart notebook](examples/beams_quickstart.ipynb)
- Fitting, subtracting and/or inpainting sources and clusters from a map
  — see the [clusters quickstart notebook](examples/clusters_quickstart.ipynb)
- Estimating a jitter beam through a harmonic-space cross-correlation of the map with a catalog (e.g. quasars)

Runnable, pre-executed notebooks live in [`examples/`](examples/README.md); each is
generated from a plain-Python builder script beside it.


Development
-----------

```bash
pytest              # run the test suite
ruff check .        # lint
ruff format .       # format
```

Documentation is built with Sphinx and hosted on
[Read the Docs](https://soma.readthedocs.io):

```bash
sphinx-build -b html docs docs/_build/html
```
