"""Turn a Planck dust map into the sky backdrop that soma.io draws globes on.

    python -m soma.scripts.build_backdrop COM_CompMap_dust-commander_0256_R2.00.fits

Reads the all-sky HEALPix map, rotates it Galactic -> Equatorial and resamples
it onto a full-sky CAR grid at ``--res`` degrees. The rotation is done in
spherical harmonics, so the band limit set by ``--res`` *is* the downgrade;
no aliasing of the small scales that the globe could never show anyway.

Write the result to src/soma/planck_dust_equ.fits to ship it with the package;
see the packaging notes in docs/.
"""

import argparse

import healpy as hp
import numpy as np
from pixell import enmap, reproject, utils


def make_backdrop(path, res=0.5, field=0):
    """Planck HEALPix map (Galactic) -> full-sky CAR enmap (Equatorial)."""
    hmap = np.asarray(hp.read_map(path, field=field), dtype=np.float64)  # -> RING
    hmap[~np.isfinite(hmap)] = 0.0
    hmap[hmap < -1e29] = 0.0  # HEALPix UNSEEN
    nside = hp.npix2nside(hmap.size)

    shape, wcs = enmap.fullsky_geometry(res=res * utils.degree, proj="car")
    lmax = int(min(3 * nside - 1, 180.0 / res))  # matched to the output grid
    omap = reproject.healpix2map(hmap, shape, wcs, rot="gal,equ", lmax=lmax)
    return enmap.enmap(np.asarray(omap, dtype=np.float32), wcs), nside, lmax


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="Planck all-sky HEALPix dust map (FITS)")
    ap.add_argument("-o", "--output", default="planck_dust_equ.fits")
    ap.add_argument(
        "-r", "--res", type=float, default=0.5, help="output resolution in degrees (default: 0.5)"
    )
    ap.add_argument(
        "-f", "--field", type=int, default=0, help="column of the HEALPix file to read (default: 0)"
    )
    a = ap.parse_args()

    omap, nside, lmax = make_backdrop(a.input, res=a.res, field=a.field)
    enmap.write_map(a.output, omap)
    print(
        f"nside {nside} -> {omap.shape[0]}x{omap.shape[1]} CAR at {a.res} deg "
        f"(lmax {lmax}), equatorial"
    )
    print(f"wrote {a.output} ({omap.nbytes / 1e6:.1f} MB)")
