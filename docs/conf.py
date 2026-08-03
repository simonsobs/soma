"""Sphinx configuration for the soma documentation."""

import re
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Allow building from a checkout in which soma has not been installed.
sys.path.insert(0, str(_ROOT / "src"))


def _release():
    """Return the full version string of the documented package.

    Uses the installed distribution metadata when available and falls back to
    parsing pyproject.toml, so that the docs can also be built from a checkout
    in which soma has not been installed.
    """
    try:
        return get_version("soma")
    except PackageNotFoundError:
        pyproject = (_ROOT / "pyproject.toml").read_text()
        match = re.search(r'^version = "(.+?)"', pyproject, re.MULTILINE)
        return match.group(1) if match else "0.0.0"


project = "soma"
author = "Mathew Madhavacheril"
copyright = "2026, Simons Observatory Collaboration"
release = _release()
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_rtd_theme",
]

exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = f"soma {release}"

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
# pixell and healpy are compiled and slow to install, and are not needed to
# document soma's own API. numpy, scipy and yaml are left unmocked so that type
# references in signatures keep resolving through intersphinx.
autodoc_mock_imports = ["pixell", "healpy"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]
