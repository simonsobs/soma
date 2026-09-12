# Contributing

## Development environment

```bash
git clone https://github.com/simonsobs/soma.git
cd soma
pip install -e ".[dev]"
```

## Checks run by CI

The `CI` workflow (`.github/workflows/ci.yml`) runs on every push to `main` and
on every pull request. It has four jobs, all of which must pass:

1. **lint** — `ruff check .` and `ruff format --check .`
2. **test** — `pytest` on Linux and macOS for Python 3.10 through 3.13
3. **docs** — `make -C docs html`, which fails on any Sphinx warning
4. **build** — `python -m build` followed by `twine check dist/*`

Run the same checks locally before opening a pull request:

```bash
ruff check .
ruff format .
pytest
python -m build && twine check dist/*
```

## Documentation

The documentation is written in Markdown (MyST) under `docs/`. The API reference is
generated from the docstrings, and the notebooks in `examples/` are rendered with the
outputs they were saved with, so they are not executed during the build. To add a module
to the API reference, list it in `docs/api.md`.

```bash
pip install -e ".[docs]"
make -C docs html        # writes docs/_build/html/index.html
make -C docs livehtml    # serves the docs locally and rebuilds on every change
make -C docs clean
```

`make html` passes `-W --keep-going`, which reports every warning and then fails, as the
Read the Docs build does. Without make, the equivalent is

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Read the Docs builds with `fail_on_warning: true`, so a Sphinx warning fails the
docs build. Fix warnings rather than ignoring them.
