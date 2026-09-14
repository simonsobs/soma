soma
====

[![CI](https://github.com/msyriac/soma/actions/workflows/ci.yml/badge.svg)](https://github.com/msyriac/soma/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/pysoma/badge/?version=latest)](https://pysoma.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/pysoma.svg)](https://pypi.org/project/pysoma/)

[`soma`](https://en.wikipedia.org/wiki/Soma_(Brave_New_World)) (Simons Observatory Map-based Analysis) is a set of companion tools for use with the [`pixell`](https://github.com/simonsobs/pixell) library.

_This library is in active early development and its API will likely change significantly. It is meant to replace `msyriac/orphics` and integrate in modules from other libraries that supplement `pixell`._

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
[uv] pip install -e ".[docs]"
make -C docs html        # open docs/_build/html/index.html
make -C docs livehtml    # live preview with automatic rebuilds
```


Attribution
-----------

`soma` bundles one external data product: `src/soma/planck_dust_equ.fits`, the
sky backdrop that `soma.io.plot_footprints` paints its globes with. It is a
*derived* version of the Planck Commander thermal-dust map, rotated from
Galactic to equatorial coordinates, band-limited and resampled onto a 0.5° CAR
grid (see [the footprints docs](docs/footprints.md)).

If you publish a figure made with it, please cite the source map:

> Planck Collaboration X, 2016, *Planck 2015 results. X. Diffuse component
> separation: Foreground maps*, A&A **594**, A10 (arXiv:1502.01588)

and include the Planck acknowledgement:

> Based on observations obtained with Planck (http://www.esa.int/Planck), an ESA
> science mission with instruments and contributions directly funded by ESA
> Member States, NASA, and Canada.

The exact product is `COM_CompMap_dust-commander_0256_R2.00.fits`, column 0
(`I_ML` — dust intensity at 545 GHz in µK_RJ, HEALPix Nside 256, 60′ FWHM,
`PROCVER = DX11D`), from the [Planck Legacy Archive](http://pla.esac.esa.int),
also mirrored [at IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/).
