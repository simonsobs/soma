"""soma: a support library for Simons Observatory Maps and Analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("somapy")
except PackageNotFoundError:  # pragma: no cover - package is not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
