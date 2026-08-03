# Contributing

## Development environment

```bash
git clone https://github.com/simonsobs/soma.git
cd soma
pip install -e ".[dev]"
```

## Checks run by CI

The `CI` workflow (`.github/workflows/ci.yml`) runs on every push to `main` and
on every pull request. It has three jobs, all of which must pass:

1. **lint** — `ruff check .` and `ruff format --check .`
2. **test** — `pytest` on Linux and macOS for Python 3.10 through 3.13
3. **build** — `python -m build` followed by `twine check dist/*`

Run the same checks locally before opening a pull request:

```bash
ruff check .
ruff format .
pytest
python -m build && twine check dist/*
```

## Documentation

```bash
sphinx-build -b html docs docs/_build/html
```

Read the Docs builds with `fail_on_warning: true`, so a Sphinx warning fails the
docs build. Fix warnings rather than ignoring them.
