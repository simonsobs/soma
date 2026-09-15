# Installation

`soma` requires Python 3.10 or newer.

## From PyPI

```bash
pip install somapy
```

This pulls in the runtime dependencies: `numpy`, `scipy`, `pyyaml`,
`astropy`, [`pixell`](https://pixell.readthedocs.io) and
[`healpy`](https://healpy.readthedocs.io). Binary wheels exist for `pixell` and
`healpy` on Linux and macOS; on other platforms they are built from source and
require a working C/Fortran toolchain.

## From source

```bash
git clone https://github.com/simonsobs/soma.git
cd soma
pip install -e ".[dev]"
```

The optional dependency groups are:

`test`
: `pytest` and `pytest-cov` for running the test suite.

`docs`
: `sphinx`, the `furo` theme, `myst-nb`, `sphinx-copybutton` and `sphinx-autobuild`
  for building this documentation.

`dev`
: everything above, plus `ruff` and `build`.
