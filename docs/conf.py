"""Sphinx configuration for the soma documentation."""

import re
import shutil
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path

_DOCS = Path(__file__).resolve().parent
_ROOT = _DOCS.parent

# Document the source tree next to this file, even if another copy of soma is installed.
sys.path.insert(0, str(_ROOT / "src"))


def _release():
    """Return the version from the installed metadata, or from pyproject.toml."""
    try:
        return get_version("soma")
    except PackageNotFoundError:
        pyproject = (_ROOT / "pyproject.toml").read_text()
        match = re.search(r'^version = "(.+?)"', pyproject, re.MULTILINE)
        return match.group(1) if match else "0.0.0"


def _copy_examples():
    """Copy the pre-executed notebooks in examples/ into docs/examples/.

    Sphinx only reads files below docs/, and the notebooks live next to the scripts that
    build them. The copies are ignored by git.
    """
    dest = _DOCS / "examples"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir()
    for nb in sorted((_ROOT / "examples").glob("*.ipynb")):
        shutil.copy2(nb, dest / nb.name)


_copy_examples()

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
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_nb",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"soma {release}"

# API reference: api.md lists the modules and autosummary writes one page for each.
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# The notebooks are committed with their outputs and rendered as they are: running them
# needs data downloads and, for some, a GPU.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3
