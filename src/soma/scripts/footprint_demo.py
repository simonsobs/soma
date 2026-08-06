"""Demo: build a few hit maps in memory and spin them on a globe.

    python -m soma.scripts.footprint_demo [output.gif]

No data files needed; the maps are generated here.
"""

import sys

import numpy as np
from pixell import enmap, utils

from soma.io import plot_footprints


def hits(dec_range, ra_range, res=10.0, taper=4.0, holes=0, seed=0):
    """A hit map: a declination band with edges that thin out the way a scan's
    coverage does, and optional holes where sources have been cut."""
    box = np.array([[dec_range[0] - taper, ra_range[0]],
                    [dec_range[1] + taper, ra_range[1]]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj="car")
    dec, ra = enmap.posmap(shape, wcs) / utils.degree

    nhit = (np.clip((dec - dec_range[0] + taper) / taper, 0, 1)
            * np.clip((dec_range[1] + taper - dec) / taper, 0, 1))
    rng = np.random.default_rng(seed)
    for _ in range(holes):
        d0, r0 = rng.uniform(*dec_range), rng.uniform(*ra_range)
        nhit[np.hypot(dec - d0, (ra - r0) * np.cos(np.radians(d0)))
             < rng.uniform(1.5, 4.0)] = 0
    return enmap.enmap(np.round(nhit * 500), wcs)


def cap(centre, radius, res=12.0):
    """A circular patch centred on ``centre=(dec, ra)``."""
    dec0, ra0 = centre
    pad = (radius + 3) / max(np.cos(np.radians(dec0)), 0.2)
    box = np.array([[dec0 - radius - 3, ra0 - pad],
                    [dec0 + radius + 3, ra0 + pad]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj="car")
    pos = enmap.posmap(shape, wcs)[::-1].reshape(2, -1)
    r = utils.angdist(np.array([ra0, dec0]) * utils.degree, pos).reshape(shape)
    return enmap.enmap(np.round(np.clip((radius - r / utils.degree) / 2, 0, 1) * 800), wcs)


if __name__ == "__main__":
    maps = {
        "Wide survey": hits((-62, 22), (-180, 180), holes=14, seed=3),
        "Deep field": hits((-8, 8), (-40, 45), res=8, taper=2),
        "Southern cap": cap((-70, 120), 22.0),
    }
    for name, m in maps.items():
        print(f"{name}: {m.shape} pixels, max {m.max():.0f} hits")

    out = sys.argv[1] if len(sys.argv) > 1 else "demo.gif"
    # Default framing and speed; just a smaller globe, to keep the GIF small.
    areas = plot_footprints(maps, output=out, size=560,
                            title="Survey footprints", galactic_plane=True)
    print("\n".join(f"{k}: {v:,.0f} deg^2" for k, v in areas.items()))
    print("wrote", out)
