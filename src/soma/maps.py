"""
Map-level helpers: geometry, centroiding, recentring, painting objects
onto a map, masks, and model maps of SZ clusters and point sources
(predict amplitudes from an Arnaud pressure profile or fit them on a map,
paint the model, and stack before/after -- see :func:`build_object_model`), and
constrained-Gaussian inpainting of holes at catalog positions
(see :class:`Inpainter`).
"""

import time

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from pixell import coordinates, enmap, pointsrcs, reproject, utils, wcsutils
from pixell import fft as enfft
from pixell import mpi as pmpi
from scipy import integrate, special
from scipy.interpolate import interp1d
from scipy.linalg import LinAlgError, cho_factor, cho_solve, eigh
from scipy.special import j0

from .theory import xi_from_cl

__all__ = [
    "check_enmap",
    "real_grid",
    "centroid_pixels",
    "fourier_recenter",
    "paint_profile",
    "paint_sources",
    "paint_elliptical",
    "paint_objects",
    "stack_thumbnails",
    "fit_object_amplitudes",
    "stamp_cov",
    "make_geometry",
    "inpaint_stamp",
    "Inpainter",
    "deltaT_profile",
    "m200c_to_m500c",
    "point_source_profile",
    "BeamProfiler",
    "build_object_model",
    "COSMO",
    "cosine_apodize",
    "wfactor",
]


def check_enmap(bmap, ncomp=None):
    """Raise unless `bmap` is a pixell enmap of the expected shape.
    `ncomp` optionally requires a leading component axis of that length
    (3 for a T, Q, U stack).
    """
    if not isinstance(bmap, enmap.ndmap):
        raise TypeError(f"expected a pixell enmap, got {type(bmap).__name__}.")
    want = 2 if ncomp is None else 3
    if bmap.ndim != want or (ncomp is not None and bmap.shape[0] != ncomp):
        shp = "(Ny, Nx)" if ncomp is None else f"({ncomp}, Ny, Nx)"
        raise ValueError(f"expected {shp}, got {tuple(bmap.shape)}")


def real_grid(shape, wcs, center=None):
    """Flat-sky offsets (dy, dx) in radians from `center`, per pixel.
    Flat on purpose (unlike pixell.enmap.posmap) so that it can be used with other
    functions that use FFTs.
    """
    ny, nx = shape[-2:]
    cy, cx = (ny // 2, nx // 2) if center is None else np.asarray(center, dtype=float)
    dpix = enmap.pixshape(shape[-2:], wcs, signed=True)
    y = (np.arange(ny) - cy) * dpix[0]
    x = (np.arange(nx) - cx) * dpix[1]
    return np.meshgrid(y, x, indexing="ij")


def centroid_pixels(imap, wrap_safe=True, thresh=0.0):
    """First-moment centroid (y0, x0) of a 2D map, in pixels.

    wrap_safe=True first rolls the peak to the array center, so that
    something straddling the periodic map edge (e.g. stored with its peak
    at pixel (0, 0)) is handled correctly; the returned coordinates are
    mapped back to the original frame afterwards, and may be non-integer
    and, for wrapped input, outside [0, N).

    `thresh` ignores everything below that fraction of the peak, by
    subtracting the level and clipping at zero. 
    """
    b = np.asarray(imap, dtype=float)
    if b.ndim != 2:
        raise ValueError("centroid_pixels expects a 2D map")
    ny, nx = b.shape
    if wrap_safe:
        iy, ix = np.unravel_index(np.nanargmax(b), b.shape)
        sy, sx = ny // 2 - iy, nx // 2 - ix
        b = np.roll(b, (sy, sx), axis=(-2, -1))
    else:
        sy = sx = 0
    if thresh:
        b = np.clip(b - thresh * np.nanmax(b), 0.0, None)
    tot = b.sum()
    if tot == 0:
        raise ValueError("map sums to zero; centroid undefined")
    gy, gx = np.mgrid[0:ny, 0:nx]
    return float((gy * b).sum() / tot - sy), float((gx * b).sum() / tot - sx)


def fourier_recenter(imap, center, fourier=False):
    """Move whatever sits at pixel `center` to the origin, exactly.
    Thin wrapper around pixell.fft.shift.
    
    Parameters
    ----------
    imap : (..., Ny, Nx) real-space map, or its *natural-order*
        (un-fftshifted) transform when `fourier=True`. An enmap keeps its
        geometry and any leading component axes come along untouched.
    center : (y0, x0) pixel position, which may be fractional. Whatever
        is there ends up at pixel (0, 0). To translate content *by* an
        offset d instead, pass ``center=-d``.
    fourier : the input has already been transformed, so apply the phase
        ramp and stop rather than doing fft -> phase -> ifft. 

    """
    y0, x0 = np.asarray(center, dtype=float)
    return enfft.shift(imap, [-y0, -x0], nofft=fourier)


# -------------------------------------------------------------------------
# painting objects
# -------------------------------------------------------------------------
def paint_profile(fwhm_arcmin, npoints=10000):
    """Build a unit-peak Gaussian painting profile for pointsrcs.sim_objects.

    Parameters
    ----------
    fwhm_arcmin : float
        FWHM of the painting kernel in arcminutes.
    npoints : int
        Number of radial samples.

    Returns
    -------
    profile : ndarray
        Array of shape (2, npoints) containing radii (radians) and the
        unit-peak Gaussian profile evaluated at those radii.
    """
    sigma = np.deg2rad(fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    r = np.linspace(0.0, 10.0 * sigma, npoints)
    return np.array([r, np.exp(-0.5 * (r / sigma) ** 2)])


def paint_sources(shape, wcs, ras_deg, decs_deg, amps, profile):
    """Paint radially symmetric objects into a map with sub-pixel accuracy.

    Parameters
    ----------
    shape, wcs : tuple, astropy wcs
        Geometry of the output map.
    ras_deg, decs_deg : ndarray
        Source coordinates in degrees.
    amps : ndarray
        Peak amplitudes of each object in map units.
    profile : ndarray
        (2, nsamp) radial profile (r in radians, unit peak).

    Returns
    -------
    omap : enmap.ndmap
        Map with the painted objects.
    """
    poss = np.array([np.deg2rad(decs_deg), np.deg2rad(ras_deg)])
    amps = np.asarray(amps, dtype=np.float32)
    vmin = np.abs(amps).max() * 1e-5
    return pointsrcs.sim_objects(shape, wcs, poss, amps, profile, vmin=vmin)


def _stamp_offsets(shape, wcs, ra0, dec0, iy, ix, half_y, half_x):
    """North/east offsets (u, v) in radians of a stamp's pixels from a source.

    The stamp is the pixel block centred on integer pixel (iy, ix) and
    running +-(half_y, half_x); the offsets are measured from the exact
    source position (ra0, dec0) in radians, so a sub-pixel offset between
    the source and the stamp centre is carried correctly.

    Works for any projection: the pixel centres are sent to the sky with
    `enmap.pix2sky`, then rotated by `coordinates.recenter` into a frame
    whose origin is the source and whose north is the source's local
    meridian. u is north and v = ra * cos(dec) is east, the frame
    `harmonic.catalog_spin_alm` measures position angles in. Nothing here
    reads a cdelt sign, so the frame cannot be mirrored by the wcs
    handedness.

    Pixel indices outside the map are expected and fine: `pix2sky` uses
    `wcsutils.nobcheck`, so it is a plain affine plus projection
    evaluation, and `recenter` goes through the 3-vector rotation, which
    is exactly 2*pi periodic in ra and continues correctly past the pole.
    Hence `safe=False`: the RA-cut unwinding would be pure cost.

    (u, v) is the local sinusoidal frame rather than the exactly
    gnomonic (theta cos phi, theta sin phi); the two differ at O(theta^3),
    which is 0.04 arcsec at 25 arcmin from the source.
    """
    dy, dx = np.mgrid[-half_y : half_y + 1, -half_x : half_x + 1]
    pix = np.array([iy + dy, ix + dx], dtype=float)
    dec, ra = enmap.pix2sky(shape[-2:], wcs, pix, safe=False)
    ra_r, dec_r = coordinates.recenter(np.array([ra, dec]), [ra0, dec0, 0.0, 0.0])
    return dec_r, ra_r * np.cos(dec_r)


def paint_elliptical(
    shape,
    wcs,
    ras_deg,
    decs_deg,
    amps,
    fwhm_maj_arcmin,
    fwhm_min_arcmin,
    pa_deg=0.0,
    nsigma=5.0,
    method="auto",
):
    """Paint elliptical Gaussians with a fixed local-meridian position angle.

    pixell's pointsrcs.sim_objects only handles radially symmetric
    profiles, so anisotropic objects (needed to inject an m=2 signal into
    a simulation) are painted here with a per-source pixel stamp. The
    ellipse orientation is defined in the local meridian frame: the
    position angle is measured from local north towards local east, which
    is the same frame the unoriented multipole estimator uses.

    Two ways of building that frame:

    - the fast path, for CAR and for "plain" geometries, where the north
      and east offsets of a pixel are just its index offsets times the
      *signed* cdelt, the east one shrunk by cos(dec) at the source. It
      is a flat frame pinned at the source's declination, so it drifts
      across a tall stamp at high declination.
    - the general path, which sends each stamp pixel through `pix2sky`
      and rotates it into the source's frame (see `_stamp_offsets`). It
      is correct for any projection and any declination, and costs one
      extra projection evaluation per source.

    Parameters
    ----------
    shape, wcs : tuple, astropy wcs
        Geometry of the output map. Any projection; only the last two
        axes of `shape` are used.
    ras_deg, decs_deg : ndarray
        Source coordinates in degrees.
    amps : ndarray
        Peak amplitudes of each object in map units.
    fwhm_maj_arcmin, fwhm_min_arcmin : float
        Major and minor axis FWHM in arcminutes.
    pa_deg : float
        Position angle of the major axis, degrees east of north.
    nsigma : float
        Stamp half-size in units of the major-axis sigma.
    method : str
        "auto" (default) takes the fast path for CAR and plain
        geometries and the general one otherwise; "car" and "general"
        force one or the other. Forcing "general" on a CAR map measures
        how far the flat-frame approximation is moving a result.

    Returns
    -------
    omap : enmap.ndmap
        Map with the painted objects, float32.

    Notes
    -----
    Sources whose stamp runs off the map in x wrap in RA on cylindrical
    geometries and are clipped otherwise.

    Known limitation: a source within nsigma*smaj of a pole loses the
    wings of its stamp that cross it. `insert_at` caps the rows rather
    than folding them around the pole.
    """
    if len(shape) < 2:
        raise ValueError(f"expected a 2D (or higher) geometry, got shape {tuple(shape)}")
    if wcs.naxis != 2:
        raise ValueError(f"expected a 2D wcs, got naxis={wcs.naxis}")
    if method not in ("auto", "car", "general"):
        raise ValueError(f"method must be one of 'auto', 'car', 'general'; got {method!r}")

    fac = np.sqrt(8.0 * np.log(2.0))
    smaj = np.deg2rad(fwhm_maj_arcmin / 60.0) / fac
    smin = np.deg2rad(fwhm_min_arcmin / 60.0) / fac
    rmax = nsigma * smaj

    # is_separable alone would let CEA and MER through, which are separable
    # cylindricals whose declination is not linear in y. A plain geometry
    # takes the fast path with cosd = 1: its cdelt is a literal,
    # declination-independent spacing, and a cos(dec) would be spurious.
    plain = wcsutils.is_plain(wcs)
    fast = method == "car" or (
        method == "auto"
        and (plain or (wcsutils.is_separable(wcs) and wcsutils.get_proj(wcs) == "car"))
    )
    cyl = wcsutils.is_cyl(wcs)
    nphi = utils.nint(360.0 / abs(wcs.wcs.cdelt[0])) if cyl else 0
    ny, nx = shape[-2:]
    # both caps are lossless: a stamp wider than half the sky in x, or taller
    # than the map, only re-covers sky it already covers
    xcap = min(nphi // 2, nx) if cyl else nx

    omap = enmap.zeros((ny, nx), wcs, np.float32)
    ras_rad, decs_rad = np.deg2rad(ras_deg), np.deg2rad(decs_deg)
    py, px = enmap.sky2pix((ny, nx), wcs, np.array([decs_rad, ras_rad]))
    pa = np.deg2rad(pa_deg)
    ca, sa = np.cos(pa), np.sin(pa)
    # signed, so that u really is north and v really is east whatever the
    # handedness of the wcs.
    cdy = np.deg2rad(wcs.wcs.cdelt[1])
    cdx = np.deg2rad(wcs.wcs.cdelt[0])

    for i in range(py.size):
        # a source outside the projection's domain projects to nan
        if not (np.isfinite(py[i]) and np.isfinite(px[i])):
            continue
        iy, ix = int(np.rint(py[i])), int(np.rint(px[i]))
        if fast:
            cosd = 1.0 if plain else max(np.cos(decs_rad[i]), 1e-8)
            half_y = min(int(np.ceil(rmax / abs(cdy))), ny)
            half_x = min(int(np.ceil(rmax / (abs(cdx) * cosd))), xcap)
            dy, dx = np.mgrid[-half_y : half_y + 1, -half_x : half_x + 1]
            # local meridian frame: u towards north, v towards east
            u = (dy - (py[i] - iy)) * cdy
            v = (dx - (px[i] - ix)) * cdx * cosd
        else:
            # size the stamp from the local d(u,v)/d(y,x) Jacobian. hypot of
            # a row of its inverse is exactly the half-width in that pixel
            # axis needed to contain the disc of radius rmax, and unlike a
            # per-axis scale it survives a skewed or rotated pc matrix.
            u1, v1 = _stamp_offsets(shape, wcs, ras_rad[i], decs_rad[i], iy, ix, 1, 1)
            jac = np.array(
                [
                    [(u1[2, 1] - u1[0, 1]) / 2, (u1[1, 2] - u1[1, 0]) / 2],
                    [(v1[2, 1] - v1[0, 1]) / 2, (v1[1, 2] - v1[1, 0]) / 2],
                ]
            )
            det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
            if not np.isfinite(det) or det == 0:
                continue
            inv = np.linalg.inv(jac)
            half_y = min(int(np.ceil(rmax * np.hypot(*inv[0]))), ny)
            half_x = min(int(np.ceil(rmax * np.hypot(*inv[1]))), xcap)
            u, v = _stamp_offsets(shape, wcs, ras_rad[i], decs_rad[i], iy, ix, half_y, half_x)
        a = u * ca + v * sa
        b = -u * sa + v * ca
        stamp = amps[i] * np.exp(-0.5 * ((a / smaj) ** 2 + (b / smin) ** 2))
        enmap.insert_at(
            omap,
            [[iy - half_y, ix - half_x], [iy + half_y + 1, ix + half_x + 1]],
            stamp,
            op=np.add,
            wrap="auto" if cyl else 0,
        )
    return omap


def paint_healpix_profile(nside, pos, amps, profile, oversample=3, rmax=None, nest=False):
    """Paint radially symmetric objects into a HEALPix map, pixel-averaged.

    Each output pixel gets the *average* of the profile over its area, not
    the profile sampled at its centre, so the map carries the HEALPix pixel
    window the way a real observation does. HEALPix is hierarchical, which
    makes that average cheap and exact: in NESTED ordering a pixel at
    `nside` is precisely the union of `4**oversample` pixels at
    `nside * 2**oversample`, so evaluating the profile at the centres of
    those children and averaging them is a genuine area average, converging
    as the sub-pixel spacing shrinks.


    Parameters
    ----------
    nside : int
        HEALPix resolution of the output map.
    pos : (n, 2) ndarray
        [dec, ra] in radians, as `catalogs.random_positions` returns and
        `thumbnails_healpix` consumes.
    amps : scalar or (n,) ndarray
        Peak amplitude of each object, in map units.
    profile : (2, nsamp) ndarray
        Radial profile: radii in radians and the unit-peak profile there,
        as `paint_profile` builds. Interpolated linearly, zero beyond the
        tabulated range.
    oversample : int
        Each pixel is averaged over `4**oversample` children. 0 point-samples
        the pixel centres instead, giving a map with *no* pixel window --
        the control needed to separate the window from anything downstream.
        3 (64 children) is converged to well under a per-cent for a profile
        resolved by the pixels.
    rmax : float or None
        Painting radius in radians. Defaults to the largest tabulated
        profile radius. Pixels beyond it get nothing, so it must reach far
        enough out that the profile is negligible there.
    nest : bool
        Ordering of the returned map. The internal work is NESTED either
        way; RING costs one reorder.

    Returns
    -------
    hmap : (npix,) ndarray
        The painted map. Overlapping objects add.
    """
    pos = np.atleast_2d(np.asarray(pos, dtype=float))
    amps = np.broadcast_to(np.asarray(amps, dtype=float), (len(pos),))
    profile = np.asarray(profile, dtype=float)
    if rmax is None:
        rmax = float(profile[0].max())
    nhi, fac = nside * 2**oversample, 4**oversample

    parents, values = [], []
    for (dec, ra), amp in zip(pos, amps, strict=True):
        vec = hp.ang2vec(np.pi / 2 - dec, ra)
        ipix = hp.query_disc(nhi, vec, rmax, nest=True, inclusive=True)
        theta, phi = hp.pix2ang(nhi, ipix, nest=True)
        ang = hp.rotator.angdist(np.array([theta, phi]), np.array([np.pi / 2 - dec, ra]))
        parents.append(ipix // fac)
        values.append(amp * np.interp(ang, profile[0], profile[1], right=0.0))

    # divide by the full child count, not by how many landed inside rmax:
    # the children outside contribute zero to the average, not nothing
    counts = np.bincount(
        np.concatenate(parents),
        weights=np.concatenate(values),
        minlength=hp.nside2npix(nside),
    )
    hmap = counts / fac
    return hmap if nest else hp.reorder(hmap, n2r=True)


# -------------------------------------------------------------------------
# Masks
# -------------------------------------------------------------------------
def cosine_apodize(bmask, width_deg):
    """Cosine-taper a binary mask over `width_deg` from its edge.

    The taper runs as (1 - cos(pi r / R)) / 2 in the distance r from the
    nearest masked pixel, so it is 0 at the edge, 1 beyond the width, and
    smooth at both ends. `width_deg=None` returns the mask untouched.

    """
    if width_deg is None:
        return bmask
    return enmap.apod_mask(bmask, width=width_deg * utils.degree, edge=False)


def wfactor(n, mask, sht=True, pmap=None, equal_area=False):
    """
    Approximate correction to an n-point function for the loss of power
    due to the application of a mask.

    For an n-point function using SHTs, this is the ratio of
    area weighted by the nth power of the mask to the full sky area 4 pi.
    This simplifies to mean(mask**n) for equal area pixelizations like
    healpix. For SHTs on CAR, it is sum(mask**n * pixel_area_map) / 4pi.
    When using FFTs, it is the area weighted by the nth power normalized
    to the area of the map. This also simplifies to mean(mask**n)
    for equal area pixels. For CAR, it is sum(mask**n * pixel_area_map)
    / sum(pixel_area_map).

    If not, it does an expensive calculation of the map of pixel areas. If this has
    been pre-calculated, it can be provided as the pmap argument.
    """
    assert mask.ndim == 1 or mask.ndim == 2
    if pmap is None:
        if equal_area:
            npix = mask.size
            pmap = 4 * np.pi / npix if sht else enmap.area(mask.shape, mask.wcs) / npix
        else:
            pmap = enmap.pixsizemap(mask.shape, mask.wcs)
    return (
        np.sum((mask**n) * pmap) / np.pi / 4.0 if sht else np.sum((mask**n) * pmap) / np.sum(pmap)
    )


# -------------------------------------------------------------------------
# Healpix helpers
# -------------------------------------------------------------------------


def healpix_interp_window(nside, ell, nsamp=200000, nbin=1024, seed=0):
    """Transfer function of the bilinear interpolation in `thumbnails_healpix`.

    Cutting a thumbnail out of a HEALPix map reconstructs the field from the
    four nearest pixel centres (`hp.get_interp_val`). Averaged over the
    sub-pixel phase (which a stack over a random catalog does) that is a
    convolution, so a stack carries this on top of the beam and the pixel
    window:

        stack(ell) = b_ell * w_ell * healpix_interp_window

    with w_ell the HEALPix pixel window (`healpy.pixwin`).

    It is not a small correction. At nside=512 it removes 9% of the signal
    by ell=500 and 33% by ell=1000, and being a third transfer function it
    is easy to mistake for beam.

    Method
    ------
    The effective real-space kernel (the average over sub-pixel phase)
    is a sum of weighted deltas at the
    neighbour offsets. Its azimuthally averaged transform is therefore

        T(ell) = < sum_i w_i J_0(ell theta_i) >

    over directions drawn uniformly on the sphere, with theta_i the angular
    distance to neighbour i.

    This is the *amplitude* window (the mean of the kernel) which is what
    a stack of many objects measures. It is not the right correction for a
    power spectrum, which sees sqrt(<|K|^2>) instead. But if you are calculating
    a power spectrum, you probably never should be extracting thumbnails
    from a healpix map.

    Parameters
    ----------
    nside : int
        Resolution of the map being cut from.
    ell : array_like
        Multipoles to evaluate at.
    nsamp : int
        Directions used for the phase average. 2e5 converges T to ~1e-4.
    nbin : int
        The weighted separations are histogrammed before the Bessel sum, so
        the cost is nbin rather than 4*nsamp evaluations per ell. 1024 bins
        over a range of about one pixel is far finer than J_0 varies.
    seed : int
        Fixes the sampling so that the answer is reproducible.

    Returns
    -------
    w : ndarray
        Transfer function, 1 at ell = 0.
    """
    rng = np.random.default_rng(seed)
    theta = np.arccos(rng.uniform(-1.0, 1.0, nsamp))
    phi = rng.uniform(0.0, 2.0 * np.pi, nsamp)
    pix, wgt = hp.get_interp_weights(nside, theta, phi)
    ptheta, pphi = hp.pix2ang(nside, pix.ravel())
    sep = hp.rotator.angdist(
        np.array([ptheta, pphi]),
        np.array([np.tile(theta, 4), np.tile(phi, 4)]),
    )
    # collapse to a weighted histogram of separations once, so evaluating
    # more ell costs nbin Bessel calls rather than 4*nsamp
    hist, edges = np.histogram(sep, bins=nbin, weights=wgt.ravel() / nsamp)
    centres = 0.5 * (edges[1:] + edges[:-1])
    ell = np.asarray(ell, dtype=float)
    return (hist * special.j0(ell[..., None] * centres)).sum(axis=-1)


def _beam_fn(beam):
    """Return B(ell) from a FWHM float, a bl array, a callable, or None."""
    if beam is None:
        return lambda ells: np.ones_like(np.asarray(ells, dtype=float))
    if callable(beam):
        return beam
    beam_arr = np.atleast_1d(np.asarray(beam, dtype=np.float64))
    if beam_arr.size == 1:
        from . import beams as sbeams

        fwhm = float(beam_arr[0])
        return lambda ells: sbeams.gaussian_bl(ells, fwhm)
    interp = interp1d(
        np.arange(beam_arr.size, dtype=float),
        beam_arr,
        bounds_error=False,
        fill_value=(beam_arr[0], 0.0),
    )
    return lambda x: interp(x)


def paint_objects(shape, wcs, ras_deg, decs_deg, amps, profiles):
    """Paint objects with an independent radial profile per object.

    The per-object generalization of :func:`paint_sources`: each object
    carries its own (2, nsamp) unit-peak radial profile (r in radians),
    scaled by its amplitude, painted with sub-pixel accuracy.

    Parameters
    ----------
    shape, wcs : tuple, astropy wcs
        Geometry of the output map.
    ras_deg, decs_deg : ndarray
        Object coordinates in degrees.
    amps : ndarray
        Peak amplitudes in map units.
    profiles : ndarray or list
        A single (2, nsamp) profile shared by all objects, or one per
        object.

    Returns
    -------
    omap : enmap.ndmap
        Map with the painted objects.
    """
    poss = np.array([np.deg2rad(decs_deg), np.deg2rad(ras_deg)])
    amps = np.asarray(amps, dtype=np.float32)
    if isinstance(profiles, np.ndarray) and profiles.ndim == 2:
        profiles = [profiles] * len(amps)
    rmax = max(float(p[0][-1]) for p in profiles)
    return pointsrcs.sim_objects(
        shape,
        wcs,
        poss,
        amps,
        list(profiles),
        prof_ids=np.arange(len(amps)),
        rmax=rmax,
    )


def stack_thumbnails(imap, ras_deg, decs_deg, r_arcmin=10.0, res_arcmin=None):
    """Mean tangent-plane thumbnail of a map at catalog positions.

    Parameters
    ----------
    imap : enmap.ndmap
        Input map.
    ras_deg, decs_deg : ndarray
        Stack positions in degrees.
    r_arcmin : float
        Thumbnail half-width in arcminutes.
    res_arcmin : float or None
        Thumbnail resolution; defaults to the map's own pixel size.

    Returns
    -------
    stack : enmap.ndmap
        The mean thumbnail (gnomonic projection).
    """
    if res_arcmin is None:
        res_arcmin = np.abs(imap.wcs.wcs.cdelt[1]) * 60.0
    coords = np.deg2rad(np.stack([np.atleast_1d(decs_deg), np.atleast_1d(ras_deg)], axis=-1))
    thumbs = reproject.thumbnails(
        imap, coords, r=r_arcmin * utils.arcmin, res=res_arcmin * utils.arcmin, proj="tan"
    )
    return enmap.enmap(np.asarray(thumbs).mean(axis=0), thumbs.wcs)


def fit_object_amplitudes(
    imap,
    ras_deg,
    decs_deg,
    profiles,
    cl,
    beam=None,
    noise_uK_arcmin=15.0,
    ivar=None,
    radius_arcmin=16.0,
    dec_band_deg=1.0,
    jitter=1e-10,
):
    """Generalized-least-squares amplitudes of radial templates at positions.

    For each catalog object a stamp is cut and the amplitude of its radial
    template is solved by GLS against the pixel covariance C = S_cmb + N,
    built with :func:`soma.theory.xi_from_cl`'s correlation functions
    (non-periodic, CAR-correct, full large-scale variance -- so no
    explicit low-ell cut or mean deprojection is needed; the covariance
    itself downweights the modes a stamp cannot constrain).  Covariance
    factorizations are cached per (stamp shape, declination band) when the
    noise is uniform, so the per-object cost after the first few objects
    is a pair of triangular solves.

    Parameters
    ----------
    imap : enmap.ndmap or str
        Map to fit on (a path is read stamp-by-stamp, memory-light).
    ras_deg, decs_deg : ndarray
        Object positions in degrees.
    profiles : ndarray or list
        Radial templates in map units at unit amplitude: a single (2, n)
        array shared by all objects, one per object, or -- for a per-object
        scale/shape grid -- a list of candidate (2, n) templates per
        object, in which case the candidate minimising chi-square wins and
        its index is reported.
    cl : ndarray
        1D CMB TT spectrum of the map before the beam.
    beam : float, ndarray, callable or None
        Beam folded into the covariance (``cl * B(ell)**2``): a Gaussian
        FWHM in arcmin, a bl array, or B(ell).  The templates themselves
        must already be beam-convolved.
    noise_uK_arcmin : float
        Uniform white-noise level of the map.
    ivar : enmap or str or None
        Per-pixel inverse variance; overrides ``noise_uK_arcmin`` and
        disables covariance caching.
    radius_arcmin : float
        Half-size of the fitting stamp.
    dec_band_deg : float
        Declination-band width sharing one cached factorization.
    jitter : float
        Starting relative diagonal jitter for near-singular covariances
        (band-limited / noiseless maps).

    Returns
    -------
    dict
        ``amps``, ``errs`` and ``best`` (chosen candidate index) arrays;
        objects whose stamp contains non-finite pixels get NaN.
    """
    ras = np.atleast_1d(np.asarray(ras_deg, dtype=np.float64))
    decs = np.atleast_1d(np.asarray(decs_deg, dtype=np.float64))
    nobj = ras.size
    if isinstance(profiles, np.ndarray) and profiles.ndim == 2:
        profiles = [profiles] * nobj
    cands = [p if isinstance(p, (list, tuple)) else [p] for p in profiles]

    cl = np.asarray(cl, dtype=np.float64)
    bl = _beam_fn(beam)(np.arange(cl.size, dtype=float))
    xi = xi_from_cl(cl * bl**2, rmax_rad=6.0 * radius_arcmin * utils.arcmin)

    if isinstance(imap, str):
        shape, wcs = enmap.read_map_geometry(imap)
    else:
        shape, wcs = imap.shape, imap.wcs
    coords = np.deg2rad(np.stack([decs, ras], axis=-1))
    pixboxes = enmap.neighborhood_pixboxes(shape[-2:], wcs, coords, radius_arcmin * utils.arcmin)

    dec_band = np.deg2rad(dec_band_deg)
    cache = {}
    amps = np.full(nobj, np.nan)
    errs = np.full(nobj, np.nan)
    best = np.zeros(nobj, dtype=int)
    for i in range(nobj):
        stamp = _read_stamp(imap, pixboxes[i])
        if not np.all(np.isfinite(np.asarray(stamp))):
            continue
        istamp = _read_stamp(ivar, pixboxes[i]) if ivar is not None else None
        key = None
        if istamp is None:
            key = (tuple(stamp.shape[-2:]), int(np.round(np.deg2rad(decs[i]) / dec_band)))
        if key is not None and key in cache:
            cho = cache[key]
        else:
            cov = stamp_cov(
                stamp.shape,
                stamp.wcs,
                xi,
                ncomp=1,
                ivar=istamp,
                noise_uK_arcmin=noise_uK_arcmin if istamp is None else None,
            )
            cho = _cho_with_jitter(cov, jitter)
            if key is not None:
                cache[key] = cho
        d = np.asarray(stamp, dtype=np.float64).reshape(-1)
        cinv_d = cho_solve(cho, d)
        modr = np.asarray(stamp.modrmap())
        pick = None
        for j, prof in enumerate(cands[i]):
            t = np.interp(modr, prof[0], prof[1]).reshape(-1)
            tct = float(t @ cho_solve(cho, t))
            if tct <= 0:
                continue
            a = float(t @ cinv_d) / tct
            chi2 = -(a**2) * tct  # constant d^T C^-1 d omitted
            if pick is None or chi2 < pick[0]:
                pick = (chi2, a, 1.0 / np.sqrt(tct), j)
        if pick is not None:
            _, amps[i], errs[i], best[i] = pick
    return dict(amps=amps, errs=errs, best=best)


def thumbnails_healpix(hmap, pos, r=None, res=None, proj="tan", nest=False):
    """Extract thumbnails from a HEALPix map, centred on one or more positions.

    hmap : (npix,) or (1,npix) HEALPix map, same coordinate system as pos ('cel').
    pos  : [dec,ra] or (n,2) array of [dec,ra], radians.
    r    : half-size of the thumbnail, radians. Defaults to 10 arcmin.
    res  : pixel size, radians. Defaults to half a HEALPix pixel.
    proj : default 'tan' (gnomonic)
    nest : ordering of the input map.

    Returns an enmap (ny,nx) or (n,ny,nx), centred on (0,0) so they can be
    stacked directly. Interpolation is bilinear, which introduces an
    addition pixel window (beyond the usual Healpix pixel window), which
    can be estimated in Fourier space with maps.healpix_interp_window.
    """
    hmap = np.asarray(hmap)
    if hmap.ndim == 2 and hmap.shape[0] == 1:
        hmap = hmap[0]
    if hmap.ndim != 1:
        raise ValueError(f"expected a 1D HEALPix map, got {hmap.shape}")
    if r is None:
        r = 10 * utils.arcmin
    if res is None:
        res = hp.nside2resol(hp.npix2nside(hmap.size)) / 2

    pos = np.asarray(pos, dtype=float)
    single = pos.ndim == 1
    pos = pos.reshape(-1, 2)

    shape, wcs = enmap.thumbnail_geometry(r=r, res=res, proj=proj)
    opos = enmap.posmap(shape, wcs)[::-1].reshape(2, -1)  # [ra,dec]
    dtype = hmap.dtype if hmap.dtype.kind == "f" else np.float64
    omaps = enmap.zeros((len(pos),) + shape, wcs, dtype)

    for i, (dec0, ra0) in enumerate(pos):
        # Rotate the output positions from a frame centred on (0,0) to one
        # centred on the source, as reproject.thumbnails does.
        ra, dec = coordinates.transform("cel", ["cel", [[0, 0, ra0, dec0], False]], opos)
        omaps[i] = hp.get_interp_val(hmap, np.pi / 2 - dec, ra, nest=nest).reshape(shape)

    return omaps[0] if single else omaps


# ---------------------------------------------------------------------------
# SZ cluster / point-source model maps
# ---------------------------------------------------------------------------
_TCMB_UK = 2.726e6
_H_CGS = 6.62608e-27
_K_CGS = 1.3806488e-16
SIGMA_T_CM2 = 6.6524587158e-25  # Thomson cross section [cm^2]
ME_C2_KEV = 510.998946  # electron rest-mass energy [keV]
MPC_CM = 3.0856775815e24  # 1 Mpc in cm
P_TH_TO_PE = 1.932  # thermal/electron pressure, ionized H+He (X=0.76)
ARNAUD = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
PROFILE_MAX_ARCMIN = 40.0  # models tabulated/painted out to this radius
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)


def _g_tsz(nu_ghz):
    """Non-relativistic tSZ spectral function x coth(x/2) - 4 at nu [GHz]."""
    x = _H_CGS * (1e9 * nu_ghz) / (_K_CGS * (_TCMB_UK / 1e6))
    return x / np.tanh(x / 2.0) - 4.0


# ---------------------------------------------------------------------------
# Cluster physics (Arnaud et al. 2010 universal pressure profile)
# ---------------------------------------------------------------------------
def _cluster_scales(m500c, z, cosmo):
    """Return (R500c [Mpc], theta500 [rad], P500 [keV/cm^3]) for a cluster."""
    h70 = cosmo.H0.value / 70.0
    ez = cosmo.efunc(z)
    rho_c = cosmo.critical_density(z).to(u.Msun / u.Mpc**3).value
    r500_mpc = (3.0 * m500c / (4.0 * np.pi * 500.0 * rho_c)) ** (1.0 / 3.0)
    d_a_mpc = cosmo.angular_diameter_distance(z).to(u.Mpc).value
    theta500_rad = r500_mpc / d_a_mpc
    m_term = (m500c * h70 / 3.0e14) ** (2.0 / 3.0)
    p500 = 1.65e-3 * ez ** (8.0 / 3.0) * m_term * h70**2
    return r500_mpc, theta500_rad, p500


def _gnfw_pressure(x):
    """Dimensionless Arnaud GNFW pressure profile p(x), x = r / R500c."""
    p0, c500 = ARNAUD["P0"], ARNAUD["c500"]
    g, a, b = ARNAUD["gamma"], ARNAUD["alpha"], ARNAUD["beta"]
    cx = c500 * np.asarray(x)
    return p0 / (cx**g * (1.0 + cx**a) ** ((b - g) / a))


def deltaT_profile(m500c, z, cosmo, nu_ghz, ntheta=256, los_max_r500=5.0, los_npts=400):
    """Interpolator of the tSZ decrement deltaT(theta) [uK] for one cluster.

    Integrates the Arnaud GNFW pressure along the line of sight to a
    Compton-y profile and converts to temperature at ``nu_ghz`` with the
    non-relativistic spectral function.

    Parameters
    ----------
    m500c, z : float
        Cluster mass (M500c, Msun) and redshift.
    cosmo : astropy cosmology
        E.g. the module-level ``COSMO``.
    nu_ghz : float
        Observation frequency in GHz.
    ntheta, los_max_r500, los_npts : int, float, int
        Angular samples, half line-of-sight length in R500c, LOS samples.

    Returns
    -------
    interp : callable
        deltaT(theta_rad) in uK; flat beyond its tabulated range.
    theta500_rad : float
        Angular R500c.
    """
    r500_mpc, theta500_rad, p500 = _cluster_scales(m500c, z, cosmo)
    d_a_mpc = r500_mpc / theta500_rad
    theta_grid = np.linspace(0.0, 12.0 * theta500_rad, ntheta)
    los = np.linspace(0.0, los_max_r500 * r500_mpc, los_npts)
    rperp_mpc = d_a_mpc * theta_grid
    r3d = np.sqrt(los[None, :] ** 2 + rperp_mpc[:, None] ** 2)
    x = np.maximum(r3d / r500_mpc, 1e-4)  # floor avoids the GNFW central cusp
    pe = p500 * _gnfw_pressure(x) / P_TH_TO_PE
    y_theta = (SIGMA_T_CM2 / ME_C2_KEV) * 2.0 * integrate.trapezoid(pe, los * MPC_CM, axis=1)
    dT = _g_tsz(nu_ghz) * _TCMB_UK * y_theta
    interp = interp1d(theta_grid, dT, kind="cubic", bounds_error=False, fill_value=(dT[0], dT[-1]))
    return interp, theta500_rad


def m200c_to_m500c(m200c, z, cosmo):
    """Convert M200c to M500c for an NFW profile with Duffy et al. 2008 c(M,z).

    Both overdensities are critical, so only the concentration matters;
    accurate to a few percent (well below cluster-to-cluster scatter).
    """
    h = cosmo.H0.value / 100.0

    def mu(x):
        return np.log(1.0 + x) - x / (1.0 + x)

    with np.errstate(divide="ignore", invalid="ignore"):
        c200 = 5.71 * (m200c * h / 2.0e12) ** -0.084 * (1.0 + np.asarray(z)) ** -0.47
        lo = np.full_like(c200, 0.2)
        hi = np.full_like(c200, 1.0)
        for _ in range(60):
            y = 0.5 * (lo + hi)
            pos = 2.5 * y**3 * mu(c200) - mu(y * c200) > 0
            hi = np.where(pos, y, hi)
            lo = np.where(pos, lo, y)
        y = 0.5 * (lo + hi)
    return m200c * 2.5 * y**3


# ---------------------------------------------------------------------------
# Beam-convolved templates
# ---------------------------------------------------------------------------
def point_source_profile(beam_fn, rmax_rad, nr=512, lmax=30000):
    """Peak-normalised radial image of a point source through the beam.

    Uses ``pixell.utils.beam_transform_to_profile`` (exact Legendre sum on
    the sphere).  Returns (r [rad], profile) with profile(0) = 1 so a peak
    amplitude in uK multiplies it directly.
    """
    r = np.linspace(0.0, rmax_rad, nr)
    b = utils.beam_transform_to_profile(beam_fn(np.arange(lmax + 1.0)), r)
    return r, b / b[0]


class BeamProfiler:
    """Beam-convolve radial cluster profiles in 1D, free of FFT ringing.

    Encodes the beam as a radial smoothing operator K so that
    ``profile_beamed(r_i) = sum_j K[i,j] profile_raw(r_j)`` (flat-sky
    Hankel pair); beam-convolving any tabulated radial profile is then a
    single matrix-vector product.  (``pixell.utils.RadialFourierTransform``
    does not apply here: its log-radius grid and padding require decaying
    inputs.)
    """

    def __init__(self, beam_fn, rmax_rad, nr=512, lmax=30000, nl=4000):
        """Precompute K on the linear radial grid ``self.r`` up to rmax_rad."""
        self.r = np.linspace(0.0, rmax_rad, nr)
        dr = self.r[1] - self.r[0]
        ell = np.linspace(0.0, lmax, nl)
        dl = ell[1] - ell[0]
        j0r = j0(np.outer(self.r, ell))  # (nr, nl)
        self.K = (j0r * (beam_fn(ell) * ell * dl)) @ j0r.T * (self.r * dr)

    def beamed(self, dT_interp, scale):
        """Return the beam-convolved profile of ``dT_interp(theta/scale)``."""
        return self.K @ dT_interp(self.r / scale)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _gather_lists(comm, local):
    """Gather per-rank result lists onto rank 0 (single-rank/fake-comm safe)."""
    if comm.size == 1:
        return local
    gathered = comm.gather(local, root=0)
    return [x for sub in gathered for x in sub] if comm.rank == 0 else None


def _read_stamp(imap, pixbox):
    """Extract a stamp from an enmap or (memory-light) from a FITS path."""
    out = (
        enmap.read_map(imap, pixbox=pixbox)
        if isinstance(imap, str)
        else imap.extract_pixbox(pixbox)
    )
    return out[0] if out.ndim > 2 else out


def _thumb(stamp, ra_deg, dec_deg, r_rad, res_rad):
    """Tangent-plane thumbnail around (ra, dec), or None on failure."""
    pos = np.array([[np.deg2rad(dec_deg), np.deg2rad(ra_deg)]])
    try:
        t = reproject.thumbnails(stamp, pos, r=r_rad, res=res_rad, proj="tan")
        return np.asarray(t[0] if t.ndim == 3 else t)
    except Exception:
        return None


class _StackAccum:
    """Accumulate mean before/after tangent-plane stacks for one category."""

    def __init__(self):
        self.before, self.after, self.n = 0.0, 0.0, 0

    def add(self, before, after):
        """Add one object's before/after thumbnails (either may be None)."""
        if before is None or after is None:
            return
        if self.n and np.shape(before) != np.shape(self.before):
            return
        self.before, self.after = self.before + before, self.after + after
        self.n += 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_object_model(
    shape,
    wcs,
    clusters=None,
    sources=None,
    beam=1.5,
    freq_ghz=150.0,
    comm=None,
    amp_mode="catalog",
    fit_map=None,
    cl=None,
    noise=15.0,
    snr_min=0.0,
    s_grid=(0.5, 2.0, 7),
    radius_arcmin=16.0,
    stack_map=None,
    stack_arcmin=10.0,
    stack_res_arcmin=0.5,
    dec_band_deg=1.0,
    cosmo=COSMO,
    verbose=True,
):
    """Build the beam-convolved cluster + point-source model map.

    Work is distributed over the ranks of ``comm``; the painted model map
    and the diagnostics are returned on rank 0 (``None`` elsewhere).

    Parameters
    ----------
    shape, wcs :
        Geometry of the output model map.
    clusters : dict or None
        ``ras, decs`` (deg), ``zs``, ``m500c`` (Msun) arrays.
    sources : dict or None
        ``ras, decs`` (deg) and, for catalog mode, ``amps`` (peak uK).
    beam : float, ndarray or callable
        Gaussian FWHM in arcmin, a bl array (e.g. from
        ``soma.beams.read_beam``), or a callable B(ell).
    freq_ghz : float
        Frequency for the tSZ spectral function.
    comm : MPI communicator or None
        Defaults to ``pixell.mpi.COMM_WORLD`` (a fake single-rank
        communicator without mpi4py).
    amp_mode : str
        ``catalog`` (default) or ``fit``; see the module docstring.
    fit_map : enmap or str or None
        Map to fit on (required for ``fit``); also the stack default.
    cl : ndarray or None
        1D CMB TT spectrum of the map (before the beam), required for
        ``fit`` -- passed to :func:`soma.maps.fit_object_amplitudes`.
    noise : float, enmap or str
        White-noise level in uK-arcmin, or an ivar map/path (fit mode; an
        ivar map disables covariance caching).
    snr_min : float
        Minimum fit S/N to keep an object's model (fit mode).
    s_grid : (min, max, n)
        Cluster angular-scale grid (fit mode).
    radius_arcmin : float
        Half-size of the fitting stamp.
    stack_map : enmap or str or None
        Map for the mean before/after stacks (default ``fit_map``; no
        stacks if both are None).  "after" removes each object's own model
        from its stamp; neighbours remain.
    stack_arcmin, stack_res_arcmin : float
        Half-size and resolution of the stack thumbnails (arcmin).
    dec_band_deg : float
        Declination-band width sharing one cached fit covariance.
    cosmo : astropy cosmology
        Cosmology for the cluster scales.
    verbose : bool
        Rank-0 progress printing.

    Returns
    -------
    dict
        On rank 0: ``model`` (enmap), ``stacks`` = {category: {"before",
        "after", "n"}}, ``results`` = {category: list of per-object dicts
        (idx, amp, amp_err, snr, scale, kept)}, ``nskip``.  On other ranks
        every value is None.
    """
    t0 = time.time()
    comm = comm if comm is not None else pmpi.COMM_WORLD
    rank, size = comm.rank, comm.size

    def log(msg):
        if verbose and rank == 0:
            print(f"soma.szmodel: {msg}", flush=True)

    beam_fn = _beam_fn(beam)
    rmax = PROFILE_MAX_ARCMIN * utils.arcmin
    profiler = BeamProfiler(beam_fn, rmax)
    psr, psb = point_source_profile(beam_fn, rmax, nr=profiler.r.size)

    stack_map = stack_map if stack_map is not None else fit_map
    do_stacks = stack_map is not None
    do_fit = amp_mode == "fit"
    if do_fit:
        if fit_map is None:
            raise ValueError("amp_mode='fit' requires fit_map")
        if cl is None:
            raise ValueError("amp_mode='fit' requires cl (1D CMB TT spectrum)")
        try:
            noise_level, ivar_map = float(noise), None
        except (TypeError, ValueError):
            noise_level, ivar_map = 15.0, noise
        s_vals = np.linspace(s_grid[0], s_grid[1], int(s_grid[2]))

    cats = {}
    if clusters is not None and len(clusters["ras"]):
        cats["clusters"] = clusters
    if sources is not None and len(sources["ras"]):
        cats["sources"] = sources
    log(
        f"{ {k: len(v['ras']) for k, v in cats.items()} } objects on {size} rank(s); "
        f"amp_mode={amp_mode}, stacks={do_stacks}"
    )
    pixboxes = {
        k: enmap.neighborhood_pixboxes(
            shape[-2:],
            wcs,
            np.deg2rad(np.stack([v["decs"], v["ras"]], axis=-1)),
            radius_arcmin * utils.arcmin,
        )
        for k, v in cats.items()
    }

    local = []
    accum = {k: _StackAccum() for k in cats}
    nskip = 0
    for kind, cat in cats.items():
        idx = np.array_split(np.arange(len(cat["ras"])), size)[rank]
        if idx.size == 0:
            continue
        # per-object templates: unit-amplitude beam-convolved profiles
        if kind == "clusters":
            cands = []
            for i in idx:
                dT, _ = deltaT_profile(cat["m500c"][i], cat["zs"][i], cosmo, freq_ghz)
                if do_fit:
                    cands.append([np.array([profiler.r, profiler.beamed(dT, s)]) for s in s_vals])
                else:
                    cands.append(np.array([profiler.r, profiler.beamed(dT, 1.0)]))
        else:
            cands = [np.array([psr, psb])] * idx.size

        if do_fit:
            fit = fit_object_amplitudes(
                fit_map,
                cat["ras"][idx],
                cat["decs"][idx],
                cands,
                cl,
                beam=beam,
                noise_uK_arcmin=noise_level,
                ivar=ivar_map,
                radius_arcmin=radius_arcmin,
                dec_band_deg=dec_band_deg,
            )
        for j, i in enumerate(idx):
            res = dict(idx=int(i), scale=1.0, amp_err=0.0, snr=np.inf, kept=True)
            if do_fit:
                a, e, b = fit["amps"][j], fit["errs"][j], int(fit["best"][j])
                if not np.isfinite(a):
                    nskip += 1
                    continue
                templ = cands[j][b] if kind == "clusters" else cands[j]
                prof = a * templ[1]
                res.update(
                    amp=float(a),
                    amp_err=float(e),
                    snr=float(a / e) if e > 0 else 0.0,
                    scale=float(s_vals[b]) if kind == "clusters" else 1.0,
                )
                if res["snr"] < snr_min:
                    res["kept"] = False
            elif kind == "clusters":
                prof = np.asarray(cands[j][1])
                res.update(amp=1.0)
            else:
                if "amps" not in cat:
                    raise ValueError("sources need 'amps' (peak uK) in catalog mode")
                prof = cat["amps"][i] * psb
                res.update(amp=float(cat["amps"][i]))

            if do_stacks:
                try:
                    sstamp = _read_stamp(stack_map, pixboxes[kind][i])
                    if not np.all(np.isfinite(np.asarray(sstamp))):
                        raise ValueError("non-finite stamp")
                except Exception:
                    sstamp = None
                if sstamp is not None:
                    rg = profiler.r if kind == "clusters" else psr
                    own = enmap.enmap(
                        np.interp(sstamp.modrmap(), rg, prof if res["kept"] else psb * 0.0),
                        sstamp.wcs,
                    )
                    before = _thumb(
                        sstamp,
                        cat["ras"][i],
                        cat["decs"][i],
                        stack_arcmin * utils.arcmin,
                        stack_res_arcmin * utils.arcmin,
                    )
                    after = _thumb(
                        sstamp - own,
                        cat["ras"][i],
                        cat["decs"][i],
                        stack_arcmin * utils.arcmin,
                        stack_res_arcmin * utils.arcmin,
                    )
                    accum[kind].add(before, after)

            keep = res["kept"] and np.isfinite(prof[0]) and prof[0] != 0
            local.append((kind, int(i), prof if keep else None, res))

    allres = _gather_lists(comm, local)
    allstacks = _gather_lists(
        comm,
        [{k: (np.asarray(a.before), np.asarray(a.after), a.n) for k, a in accum.items()}],
    )
    nskip_all = _gather_lists(comm, [nskip])
    if rank != 0:
        return dict(model=None, stacks=None, results=None, nskip=None)

    stacks = {}
    for k in cats:
        bsum, asum, n = 0.0, 0.0, 0
        for d in allstacks:
            b, a, m = d[k]
            if m == 0 or (n and np.shape(b) != np.shape(bsum)):
                continue
            bsum, asum, n = bsum + b, asum + a, n + m
        stacks[k] = dict(before=bsum / n if n else None, after=asum / n if n else None, n=n)

    painted = [(k, i, p) for (k, i, p, r) in allres if p is not None]
    results = {k: [] for k in cats}
    for k, _i, _p, r in allres:
        results[k].append(r)
    log(
        f"painting {len(painted)} object models "
        f"(skipped {sum(nskip_all)}, gated {len(allres) - len(painted)})"
    )
    if painted:
        rgrid = {"clusters": profiler.r, "sources": psr}
        model = paint_objects(
            shape[-2:],
            wcs,
            np.array([cats[k]["ras"][i] for k, i, _ in painted]),
            np.array([cats[k]["decs"][i] for k, i, _ in painted]),
            np.array([p[0] for _, _, p in painted]),
            [np.array([rgrid[k], p / p[0]]) for k, _, p in painted],
        )
    else:
        model = enmap.zeros(shape[-2:], wcs)
    log(f"done in {time.time() - t0:.1f}s")
    return dict(model=model, stacks=stacks, results=results, nskip=sum(nskip_all))


# ---------------------------------------------------------------------------
# Constrained-Gaussian inpainting of holes at catalog positions
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Stamp pixel covariance
# ---------------------------------------------------------------------------
def stamp_cov(
    shape,
    wcs,
    xi,
    ncomp=1,
    ivar=None,
    noise_uK_arcmin=None,
    pol_noise_factor=2.0,
    iau=False,
    row_block=1024,
):
    """Build the dense pixel covariance of a stamp from correlation functions.

    Signal covariances are the tabulated correlation functions evaluated
    at the true great-circle separation of every pixel pair (CAR-correct
    at any declination, non-periodic).  For IQU the Stokes basis of each
    pair is rotated by the position angle of the separation vector, giving
    the standard flat-sky block forms

        <T T'> = xi_tt,      <T Q'> ~ xi_cross cos(2 phi),
        <Q Q'> = (xi_plus + xi_minus cos(4 phi)) / 2,   etc.

    Ordering is component-major, ``[T | Q | U]``, flat index
    ``c * Npix + y * Nx + x``.

    White noise is added on the diagonal, from a per-pixel inverse
    variance stamp (pixels with ``ivar <= 0`` get *no* noise term and
    should be excluded from the context by the caller) or from a uniform
    level converted with the local pixel areas.  Q/U noise variance is
    ``pol_noise_factor`` times temperature.

    Parameters
    ----------
    shape, wcs :
        Stamp geometry (the last two axes of shape are used).
    xi : dict
        Correlation-function interpolators from :func:`xi_from_cl`.
    ncomp : int
        1 (temperature) or 3 (IQU; requires polarization tables in xi).
    ivar : ndarray or None
        Per-pixel inverse variance of the temperature noise.
    noise_uK_arcmin : float or None
        Uniform white-noise level; ignored when ``ivar`` is given.
    pol_noise_factor : float
        Ratio of polarization to temperature noise variance.
    iau : bool
        If True, flip the sign of U (IAU polarization convention).
    row_block : int
        Covariance rows computed per block (peak-memory control).

    Returns
    -------
    ndarray
        ``(ncomp * Npix, ncomp * Npix)`` float64 covariance matrix.
    """
    if ncomp not in (1, 3):
        raise ValueError(f"ncomp must be 1 or 3, got {ncomp}")
    if ncomp == 3 and "plus" not in xi:
        raise ValueError("IQU requested but xi has no polarization tables")
    pos = enmap.posmap(shape[-2:], wcs)
    dec = np.asarray(pos[0], dtype=np.float64).reshape(-1)
    ra = np.asarray(pos[1], dtype=np.float64).reshape(-1)
    npix = dec.size
    # Block signs follow the healpix/COSMO Stokes convention of
    # pixell.curvedsky; iau=True flips the sign of U, the same
    # convention switch as pixell.enmap.queb_rotmat(..., iau=True).
    usgn = -1.0 if iau else 1.0

    cov = np.empty((ncomp * npix, ncomp * npix))
    for i0 in range(0, npix, row_block):
        sl = slice(i0, min(i0 + row_block, npix))
        dra = (ra[None, :] - ra[sl, None] + np.pi) % (2.0 * np.pi) - np.pi
        r = utils.angdist(np.array([ra[sl], dec[sl]])[:, :, None], np.array([ra, dec])[:, None, :])
        cov[sl, :npix] = xi["tt"](r)
        if ncomp == 3:
            qsl = slice(npix + sl.start, npix + sl.stop)
            usl = slice(2 * npix + sl.start, 2 * npix + sl.stop)
            dx = dra * np.cos(0.5 * (dec[sl, None] + dec[None, :]))
            dy = dec[None, :] - dec[sl, None]
            phi = np.arctan2(dy, dx)
            c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
            c4, s4 = np.cos(4 * phi), np.sin(4 * phi)
            xp, xm, xc = xi["plus"](r), xi["minus"](r), xi["cross"](r)
            tq = xc * c2
            tu = usgn * xc * s2
            qu = usgn * 0.5 * xm * s4
            # every block is symmetric in (i <-> j): the pair angle terms
            # cos/sin(2n phi) are invariant under phi -> phi + pi
            cov[sl, npix : 2 * npix] = tq
            cov[qsl, :npix] = tq
            cov[sl, 2 * npix :] = tu
            cov[usl, :npix] = tu
            cov[qsl, npix : 2 * npix] = 0.5 * (xp + xm * c4)
            cov[usl, 2 * npix :] = 0.5 * (xp - xm * c4)
            cov[qsl, 2 * npix :] = qu
            cov[usl, npix : 2 * npix] = qu

    if ivar is not None:
        var = np.zeros(npix)
        iv = np.asarray(ivar, dtype=np.float64).reshape(-1)
        good = np.isfinite(iv) & (iv > 0)
        var[good] = 1.0 / iv[good]
    elif noise_uK_arcmin is not None:
        omega = np.asarray(enmap.pixsizemap(shape[-2:], wcs)).reshape(-1)
        var = (noise_uK_arcmin * utils.arcmin) ** 2 / omega
    else:
        var = None
    if var is not None:
        idx = np.arange(npix)
        cov[idx, idx] += var
        for c in range(1, ncomp):
            cov[c * npix + idx, c * npix + idx] += pol_noise_factor * var
    return cov


# ---------------------------------------------------------------------------
# Conditional geometry and stamp filling
# ---------------------------------------------------------------------------
def _cho_with_jitter(mat, jitter, max_rel=1e-2):
    """Cholesky-factorize with escalating relative diagonal jitter.

    Band-limited or noiseless covariances can be numerically singular;
    a small multiple of the mean diagonal is added and escalated until
    the factorization succeeds, up to ``max_rel``.
    """
    scale = float(np.mean(np.diag(mat)))
    eps = jitter
    while True:
        try:
            return cho_factor(mat + (eps * scale) * np.eye(mat.shape[0]), lower=True)
        except LinAlgError:
            eps *= 100.0
            if eps > max_rel:
                raise LinAlgError(
                    "context covariance could not be factorized even with relative "
                    f"jitter {max_rel:g}; the spectra or noise model are likely "
                    "inconsistent with the map"
                ) from None


def make_geometry(
    shape,
    wcs,
    hole_radius,
    cov,
    ncomp=1,
    marginalize_mean=True,
    jitter=1e-10,
    eigval_floor=1e-8,
    valid=None,
    want_covsqrt=True,
):
    """Precompute the conditional-fill operators for one stamp geometry.

    Partitions the stamp into hole (within ``hole_radius`` of the stamp
    center) and context, optionally marginalizes over an unknown mean per
    component (a large-variance term robust against un-modelled offsets),
    and computes the conditional-mean operator and, optionally, the square
    root of the conditional covariance for constrained realizations.

    Parameters
    ----------
    shape, wcs :
        Stamp geometry.
    hole_radius : float
        Hole radius in radians, about the stamp center.
    cov : ndarray
        ``(ncomp * Npix, ncomp * Npix)`` matrix from :func:`stamp_cov`;
        not mutated.
    ncomp : int
        1 or 3.
    marginalize_mean : bool
        Marginalize over a free mean per component.
    jitter : float
        Starting relative diagonal jitter for the context factorization.
    eigval_floor : float
        Relative eigenvalue floor for the conditional covariance root.
    valid : ndarray or None
        Boolean stamp-shaped mask; False pixels are excluded from the
        context (off-map rows, ivar <= 0, non-finite data, neighbouring
        holes).  Hole pixels are always filled regardless.
    want_covsqrt : bool
        Compute the square root needed for ``add_noise`` fills.

    Returns
    -------
    dict
        ``meanmul`` (nhole x nctx), ``covsqrt`` (nhole x nhole or None),
        ``m1``/``m2`` (flat hole/context indices), ``shape``, ``ncomp``,
        ``hole_radius`` and ``cond_var`` (conditional variance diagonal).
    """
    npix = int(np.prod(shape[-2:]))
    if cov.shape != (ncomp * npix, ncomp * npix):
        raise ValueError(f"cov shape {cov.shape} does not match ncomp*npix={ncomp * npix}")
    modrmap = np.asarray(enmap.modrmap(shape[-2:], wcs)).reshape(-1)
    hole = modrmap < hole_radius
    okay = np.ones(npix, dtype=bool) if valid is None else np.asarray(valid, bool).reshape(-1)
    m1 = np.flatnonzero(np.tile(hole, ncomp))
    m2 = np.flatnonzero(np.tile(~hole & okay, ncomp))
    if m1.size == 0:
        raise ValueError("hole contains no pixels; increase hole_radius or the resolution")
    if m2.size < m1.size:
        raise ValueError(f"context ({m2.size}) smaller than hole ({m1.size}); enlarge stamp")

    if marginalize_mean:
        big = 1e4 * float(np.max(np.diag(cov)))
        cov = cov.copy()
        for c in range(ncomp):
            csl = slice(c * npix, (c + 1) * npix)
            cov[csl, csl] += big
    c11 = cov[np.ix_(m1, m1)]
    c12 = cov[np.ix_(m1, m2)]
    c22 = cov[np.ix_(m2, m2)]

    cho = _cho_with_jitter(c22, jitter)
    meanmul = cho_solve(cho, c12.T).T
    sigma = c11 - meanmul @ c12.T
    sigma = 0.5 * (sigma + sigma.T)
    covsqrt = None
    if want_covsqrt:
        evals, evecs = eigh(sigma)
        evals = np.maximum(evals, eigval_floor * max(float(evals.max()), 0.0))
        covsqrt = evecs * np.sqrt(np.maximum(evals, 0.0))[None, :]
    oshape = tuple(shape[-2:]) if ncomp == 1 else (ncomp,) + tuple(shape[-2:])
    return dict(
        meanmul=meanmul,
        covsqrt=covsqrt,
        m1=m1,
        m2=m2,
        shape=oshape,
        ncomp=ncomp,
        hole_radius=hole_radius,
        cond_var=np.diag(sigma).copy(),
    )


def inpaint_stamp(stamp, geo, add_noise=False, rng=None):
    """Fill the hole of one stamp using a precomputed geometry.

    Parameters
    ----------
    stamp : ndarray
        ``(Ny, Nx)`` or ``(ncomp, Ny, Nx)`` array matching ``geo['shape']``.
    geo : dict
        Output of :func:`make_geometry`.
    add_noise : bool
        Add a constrained-realization draw on top of the conditional mean.
    rng : numpy.random.Generator or None
        Required when ``add_noise`` is True (no hidden global state).

    Returns
    -------
    ndarray
        A filled copy of the stamp; context pixels are bit-identical.
    """
    if tuple(np.shape(stamp)) != tuple(geo["shape"]):
        raise ValueError(f"stamp shape {np.shape(stamp)} != geometry shape {geo['shape']}")
    out = np.array(stamp, dtype=np.float64, copy=True)
    flat = out.reshape(-1)
    fill = geo["meanmul"] @ flat[geo["m2"]]
    if add_noise:
        if rng is None:
            raise ValueError("add_noise=True requires a numpy Generator via rng=")
        fill = fill + geo["covsqrt"] @ rng.standard_normal(geo["m1"].size)
    flat[geo["m1"]] = fill
    return out


# ---------------------------------------------------------------------------
# Catalog driver
# ---------------------------------------------------------------------------
class Inpainter:
    """Inpaint holes at catalog positions in CAR maps, with geometry caching.

    Construct once (e.g. per MPI rank) and call on any number of maps or
    simulations sharing the same instrument model: geometries are cached
    by (stamp pixel shape, declination band) and reused across calls, so
    the dense covariance work is paid once and later maps cost one BLAS
    product per object.

    Parameters
    ----------
    cl : ndarray or dict
        Total (beam-convolved) noise-free 1D spectra of the map: a TT
        array, or a dict with ``TT`` and optionally ``EE``/``BB``/``TE``.
        For Planck-style reprojected maps pass the known/measured total
        spectrum of the map and leave the noise arguments None.
    hole_arcmin : float
        Hole radius in arcminutes.
    bl : ndarray or None
        Beam transfer applied to ``cl`` (all spectra multiplied by
        ``bl**2``), e.g. from ``soma.beams.read_beam``.
    noise_uK_arcmin : float or None
        Uniform white-noise level added to the covariance diagonal.
    ivar : enmap or None
        Full-map inverse variance of the temperature noise.  Overrides
        ``noise_uK_arcmin``, marks ``ivar <= 0`` pixels invalid, and
        disables geometry caching (each object then has its own noise).
    context_factor : float
        Stamp radius in units of the hole radius.
    marginalize_mean : bool
        Marginalize over an unknown mean per component in each stamp.
    mask_others : bool
        Exclude pixels inside other catalog objects' holes from every
        context (they hold exactly the contamination being removed).
    dec_band_deg : float
        Width of the declination bands that share a cached geometry.
    iau : bool
        IAU polarization convention (flips U).
    pol_noise_factor : float
        Q/U noise variance relative to T.
    jitter, eigval_floor : float
        Regularization controls; see :func:`make_geometry`.
    nr : int
        Radial samples for the correlation-function tables.
    """

    def __init__(
        self,
        cl,
        hole_arcmin,
        bl=None,
        noise_uK_arcmin=None,
        ivar=None,
        context_factor=3.0,
        marginalize_mean=True,
        mask_others=True,
        dec_band_deg=1.0,
        iau=False,
        pol_noise_factor=2.0,
        jitter=1e-10,
        eigval_floor=1e-8,
        nr=4096,
    ):
        if not isinstance(cl, dict):
            cl = {"TT": cl}
        cl = {k: np.array(v, dtype=np.float64) for k, v in cl.items() if v is not None}
        if bl is not None:
            bl = np.asarray(bl, dtype=np.float64)
            for k, v in cl.items():
                nl = min(v.size, bl.size)
                cl[k] = v[:nl] * bl[:nl] ** 2
        self.cl = cl
        self.pol = any(k in cl for k in ("EE", "BB", "TE"))
        self.hole_radius = hole_arcmin * utils.arcmin
        self.rtot = context_factor * self.hole_radius
        self.noise_uK_arcmin = noise_uK_arcmin
        self.ivar = ivar
        self.marginalize_mean = marginalize_mean
        self.mask_others = mask_others
        self.dec_band = np.deg2rad(dec_band_deg)
        self.iau = iau
        self.pol_noise_factor = pol_noise_factor
        self.jitter = jitter
        self.eigval_floor = eigval_floor
        self.nr = nr
        self._xi = None
        self._cache = {}

    def _xi_tables(self):
        """Build (once) the correlation tables out to beyond the stamp diagonal."""
        if self._xi is None:
            self._xi = xi_from_cl(self.cl, rmax_rad=3.0 * self.rtot, nr=self.nr)
        return self._xi

    def _geometry(self, stamp, vmask, ivar_stamp, ncomp, want_covsqrt, cache_key):
        """Build (or fetch) the conditional geometry for one stamp.

        A cached geometry lacking the constrained-realization square root
        is rebuilt (and re-cached) when ``want_covsqrt`` is requested.
        """
        if cache_key is not None and cache_key in self._cache:
            geo = self._cache[cache_key]
            if not want_covsqrt or geo["covsqrt"] is not None:
                return geo
        cov = stamp_cov(
            stamp.shape,
            stamp.wcs,
            self._xi_tables(),
            ncomp=ncomp,
            ivar=ivar_stamp,
            noise_uK_arcmin=None if ivar_stamp is not None else self.noise_uK_arcmin,
            pol_noise_factor=self.pol_noise_factor,
            iau=self.iau,
        )
        geo = make_geometry(
            stamp.shape,
            stamp.wcs,
            self.hole_radius,
            cov,
            ncomp=ncomp,
            marginalize_mean=self.marginalize_mean,
            jitter=self.jitter,
            eigval_floor=self.eigval_floor,
            valid=vmask,
            want_covsqrt=want_covsqrt,
        )
        if cache_key is not None:
            self._cache[cache_key] = geo
        return geo

    def _validity(self, stamp, pb, shape2, i, coords):
        """Context validity mask for one stamp and whether it is cacheable.

        Marks off-map rows (declination edges; RA wraps and is always
        valid), non-finite pixels, and -- with ``mask_others`` -- pixels
        inside the holes of other catalog objects.
        """
        ny = stamp.shape[-2]
        vmask = np.ones(stamp.shape[-2:], dtype=bool)
        y0, y1 = int(pb[0, 0]), int(pb[1, 0])
        if y0 < 0:
            vmask[:-y0, :] = False
        if y1 > shape2[0]:
            vmask[ny - (y1 - shape2[0]) :, :] = False
        finite = np.isfinite(np.asarray(stamp))
        while finite.ndim > 2:
            finite = finite.all(axis=0)
        vmask &= finite
        if self.mask_others and len(coords) > 1:
            d0, r0 = coords[i]
            lim = self.rtot + self.hole_radius
            ddec = np.abs(coords[:, 0] - d0)
            dra = np.abs((coords[:, 1] - r0 + np.pi) % (2 * np.pi) - np.pi) * np.cos(d0)
            near = (ddec < lim) & (dra < 1.5 * lim)
            near[i] = False
            if near.any():
                pos = enmap.posmap(stamp.shape[-2:], stamp.wcs)
                sdec = np.asarray(pos[0])
                sra = np.asarray(pos[1])
                for j in np.flatnonzero(near):
                    dj, rj = coords[j]
                    rdist = utils.angdist(np.array([sra, sdec]), np.array([rj, dj])[:, None, None])
                    vmask &= rdist >= self.hole_radius
        return vmask, bool(vmask.all())

    def __call__(
        self, imap, ras_deg, decs_deg, add_noise=False, rng=None, inplace=False, verbose=False
    ):
        """Inpaint the map at the given catalog positions.

        Parameters
        ----------
        imap : enmap
            ``(Ny, Nx)`` temperature map or ``(3, Ny, Nx)`` IQU map (the
            latter requires polarization spectra in ``cl``).
        ras_deg, decs_deg : ndarray
            Catalog positions in degrees.
        add_noise : bool
            Constrained realization instead of the mean-only fill.
        rng : numpy.random.Generator or None
            Required when ``add_noise`` is True.
        inplace : bool
            Modify ``imap`` instead of returning a filled copy.
        verbose : bool
            Print a one-line summary.

        Returns
        -------
        omap : enmap
            The inpainted map.
        info : dict
            ``n_filled``, ``n_slow`` (objects needing their own geometry
            because of invalid context pixels, neighbours or an ivar
            map), ``n_skipped`` (unusable context) and ``n_geometries``
            (cache size after the call).
        """
        if imap.ndim == 2:
            ncomp = 1
            check_enmap(imap)
        elif imap.ndim == 3 and imap.shape[0] == 3:
            ncomp = 3
            check_enmap(imap, ncomp=3)
            if not self.pol:
                raise ValueError("IQU map given but no polarization spectra in cl")
        else:
            raise ValueError(f"expected (Ny,Nx) or (3,Ny,Nx) map, got {tuple(imap.shape)}")
        ras = np.atleast_1d(np.asarray(ras_deg, dtype=np.float64))
        decs = np.atleast_1d(np.asarray(decs_deg, dtype=np.float64))
        omap = imap if inplace else imap.copy()
        shape2 = imap.shape[-2:]

        coords = np.deg2rad(np.stack([decs, ras], axis=-1))
        pixboxes = np.asarray(
            enmap.neighborhood_pixboxes(shape2, imap.wcs, coords, self.rtot), dtype=int
        )

        info = dict(n_filled=0, n_slow=0, n_skipped=0, n_geometries=0)
        for i in range(len(ras)):
            pb = pixboxes[i]
            stamp = omap.extract_pixbox(pb)
            vmask, clean = self._validity(stamp, pb, shape2, i, coords)
            ivar_stamp = None
            if self.ivar is not None:
                ivar_stamp = self.ivar.extract_pixbox(pb)
                iv = np.asarray(ivar_stamp)
                vmask &= np.isfinite(iv) & (iv > 0)
                clean = False  # per-object noise: geometry not shareable
            key = None
            if clean:
                dy, dx = pb[1] - pb[0]
                key = (ncomp, int(dy), int(dx), int(np.round(coords[i, 0] / self.dec_band)))
            else:
                info["n_slow"] += 1
            try:
                geo = self._geometry(stamp, vmask, ivar_stamp, ncomp, add_noise, key)
            except (LinAlgError, ValueError) as e:
                info["n_skipped"] += 1
                if verbose:
                    print(f"soma.maps: skipping object {i}: {e}")
                continue
            sarr = np.asarray(stamp, dtype=np.float64)
            if not np.isfinite(sarr).all():
                sarr = np.nan_to_num(sarr)  # such pixels are already outside the context
            filled = inpaint_stamp(sarr, geo, add_noise=add_noise, rng=rng)
            stamp[...] = filled
            omap = enmap.insert_at(omap, pb, stamp)
            info["n_filled"] += 1
        info["n_geometries"] = len(self._cache)
        if verbose:
            print(
                f"soma.maps: filled {info['n_filled']} (slow {info['n_slow']}, "
                f"skipped {info['n_skipped']}, geometries {info['n_geometries']})"
            )
        return omap, info
