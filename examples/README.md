Examples
========

Runnable notebooks. They need the optional plotting/notebook extras on top of
`soma` itself:

```bash
[uv] pip install -e ".[dev]" matplotlib jupyter
```

| Notebook | What it covers |
|---|---|
| [`beams_quickstart.ipynb`](beams_quickstart.ipynb) | `soma.beams` end to end: simulate a beam, decompose it into azimuthal modes $b_m(\ell)$, check both against closed forms, and read off the T→E/B leakage. |
| [`clusters_quickstart.ipynb`](clusters_quickstart.ipynb) | `soma.maps.build_object_model` end to end: simulate a sky of SZ clusters and point sources (catalog mode paints beam-convolved Arnaud templates from masses and redshifts), then fit and subtract them from the observed map (fit mode), with mean stacks before and after — plus an inpainting demo using `soma.maps.Inpainter`. |

Each notebook is generated from a plain-Python builder beside it — the source
of truth is the script, the `.ipynb` is the artifact. To regenerate after an
edit (this reformats the code cells with ruff and re-executes them, so the
committed outputs always match the code that produced them):

```bash
python examples/build_quickstart.py
python examples/build_quickstart.py --no-execute   # write only, no kernel
```
