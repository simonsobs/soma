"""Utilities involving harmonic transforms.

Features include:

- azimuthal Fourier decompositions of flat-sky images (``azimuthal_modes``,
  ``azimuthal_image`` and the metrics on their modes)
- multipole stacking on the sphere, with catalog transforms that skip pixelization
  (``catalog_spin_alm``, ``multipole_cross_spectrum``, ``beam_multipole``, ``harm2profile``)
- directional wavelet scattering covariances on the sphere (``ScatterTransform``,
  ``CARScatterTransform``, ``HealpixScatterTransform``)

Conventions
-----------

The two halves of the module measure azimuth differently:

- On the flat sky, the azimuth psi runs from +x toward +y of the map, which on a sky-frame
  map is from +RA toward +dec (``azimuthal_modes``, ``mode_orientation``).
- On the sphere, azimuth and position angle run from local north toward local east
  (``catalog_spin_alm``, ``beam_multipole``).

For a sky-frame map the two are related by PA = 90 deg - psi.

Wavelet scattering covariances
------------------------------

``ScatterTransform`` computes the statistics of ``s2scat.scatter`` (mean, variance, S1, P00,
C01 and C11, in the same order and normalisation) for a real field given its harmonic
coefficients. ``CARScatterTransform`` and ``HealpixScatterTransform`` take a CAR or a HEALPix
map instead. Each of them also takes a second field, and then returns cross statistics.

Constructing an object is the expensive step: the filters, geometries and weights of every
wavelet scale are prepared once, so reuse one object for many fields at the same band limit.

There are two backends. The numpy backend runs on the CPU. The JAX backend
(``backend="jax"``) is a jitted, differentiable function that runs wherever its input array
lives; it is optional, works in double precision, and needs pixell and ducc built with JAX
support, as described in ``docs/examples/scattering_des.ipynb``. The environment variable
``SOMA_SCATTERING_BACKEND`` sets the default: ``numpy``, ``jax``, or ``auto`` (the default),
which chooses JAX only where it runs on a GPU, since on a CPU both backends call the same
transforms.

Outline of the transform:

* The directional wavelets of s2wav, psi^j_{ln} = kappa_j(l) s_{ln}, are built
  here in numpy. For N directions only the orders
  n < N with n + N odd are non-zero; for N = 3 these are n = 0 and n = 2.
* The wavelet coefficients W^j(gamma) for the 2N-1 orientations gamma are a
  Wigner transform, which for a real field reduces to one spin-0 and one
  spin-n synthesis per active order n, combined by a fixed steering matrix
  (``ScatterTransform.wavelet``).
* The scattering covariances are computed from ``|W|`` with a second wavelet layer
  and quadrature weighted pixel sums (``ScatterTransform.from_alm``).

Typical use::

    from soma.harmonic import ScatterTransform, CARScatterTransform, HealpixScatterTransform

    st = ScatterTransform(lmax, N=3, J_min=2, mask=mask)      # input: healpy alm, any lmax
    st = CARScatterTransform(shape, wcs, mask=mask)           # input: enmap on (shape, wcs)
    st = HealpixScatterTransform(lmax, niter=3, mask=hpmask)  # input: HEALPix map, ring ordering
    mean, var, S1, P00, C01, C11 = st(x)
    mean, var, S1, P00, C01, C11 = st.from_alm(alm)           # alm directly, on any of the three
    mean, var, S1, P00, C01, C11 = st(x, y)                   # cross statistics of two fields
    jax.config.update("jax_enable_x64", True)                 # needed by the JAX backend
    st = ScatterTransform(lmax, backend="jax")                # jitted JAX function of the alm
    grad = jax.grad(lambda a: st(a)[5].sum())(alm)            # differentiable
    batched = jax.vmap(st.from_alm)                           # batches of fields
"""

import os
from math import comb, isqrt

import ducc0
import numpy as np
from pixell import curvedsky, enmap, reproject, utils, wcsutils
from scipy.ndimage import map_coordinates

from . import maps, stats

__all__ = [
    # flat sky
    "l_nyquist",
    "azimuthal_modes",
    "azimuthal_image",
    "mode_metrics",
    "mode_orientation",
    "mode_floor",
    # sphere
    "catalog_spin_alm",
    "multipole_cross_spectrum",
    "beam_multipole",
    "harm2profile",
    "analytical_tf",
    # Wavelet scattering transforms on the sphere
    "ScatterTransform",
    "CARScatterTransform",
    "HealpixScatterTransform",
]


# ---------------------------------------------------------------------------
# Fourier-grid geometry
# ---------------------------------------------------------------------------
def _ring_lmax(shape, dly, dlx):
    """Largest ell whose complete ring fits on the fftshifted grid.

    The positive-frequency side of an even-N axis stops at N/2 - 1
    samples (the +Nyquist column does not exist), so the per-axis limit
    is ``|dl| * ((N - 1) // 2)``.
    """
    ny, nx = shape[-2:]
    return float(min(abs(dly) * ((ny - 1) // 2), abs(dlx) * ((nx - 1) // 2)))


def l_nyquist(shape, wcs):
    """Largest ell at which a complete ring is representable on the grid."""
    return _ring_lmax(shape, *enmap.lpixshape(shape[-2:], wcs, signed=True))


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------
def _sample_rings(fmap, dly, dlx, ell_grid, nphi, order):
    """Spline-sample natural-order Fourier plane(s) on uniform-psi rings.

    `fmap` is (..., Ny, Nx); `dly`, `dlx` are the (signed) ell spacings
    per Fourier pixel. Returns (ok, phi, rings): `ok` flags the ell values
    whose rings fit entirely on the grid (see `_ring_lmax`), `phi` the
    nphi uniform azimuths, and `rings` a complex (..., n_ok, nphi) array
    with rings[..., i, j] = F(ell_ok[i], phi[j]). All valid rings are
    sampled in a single vectorized map_coordinates call per component
    """
    fmap = np.asarray(fmap)
    ny, nx = fmap.shape[-2:]
    phi = np.arange(nphi) * 2.0 * np.pi / nphi
    ok = np.asarray(ell_grid, float) <= _ring_lmax((ny, nx), dly, dlx)
    if not ok.any():
        return ok, phi, np.zeros(fmap.shape[:-2] + (0, nphi), complex)
    lok = np.asarray(ell_grid, float)[ok]
    # after fftshift the zero frequency sits at index N//2, so that is the
    # origin the ring is drawn about. dly/dlx are *signed*
    cc = nx // 2 + (lok[:, None] * np.cos(phi)[None, :]) / dlx  # lx index
    rr = ny // 2 + (lok[:, None] * np.sin(phi)[None, :]) / dly  # ly index
    coords = [rr.ravel(), cc.ravel()]
    # flatten any leading component axis so each Fourier plane is one pass
    planes = np.fft.fftshift(fmap, axes=(-2, -1)).reshape(-1, ny, nx)
    rings = np.empty((planes.shape[0], lok.size, nphi), complex)
    for i, p in enumerate(planes):
        rings[i] = map_coordinates(p, coords, order=order, mode="nearest").reshape(lok.size, nphi)
    return ok, phi, rings.reshape(fmap.shape[:-2] + (lok.size, nphi))


def _ring_fft(rings, nphi, mmax):
    """FFT rings (..., n_ok, nphi) in psi -> a_m of shape (..., mmax+1, n_ok).

    a_m = (1/nphi) sum_j F(psi_j) exp(-i m psi_j) is exactly a forward DFT
    with numpy's sign convention, hence fft and not ifft; the 1/nphi makes
    it the mean over the ring (a ring average) rather than a sum.
    """
    # keep only m >= 0: for a real-valued real-space map a_{-m} is
    # (-1)^m conj(a_m), so the negative half is redundant
    fm = np.fft.fft(rings, axis=-1)[..., : mmax + 1] / nphi
    # (..., n_ell, m) -> (..., m, n_ell), so a_m[i] is a spectrum in ell
    return np.swapaxes(fm, -1, -2)


def _modes_polar(fmap, dly, dlx, ell_grid, nphi, mmax, order, qu_to_eb):
    """a_m(ell) from a Fourier plane: interpolate onto uniform-psi rings
    and FFT each ring.

    With `qu_to_eb`, a (3, Ny, Nx) input is taken as T, Q, U and the
    flat-sky E/B rotation is applied on each ring before the FFT, sample
    by sample where psi is exact; the result is then (T, E, B).
    """
    ell_grid = np.asarray(ell_grid, float)
    ok, phi, rings = _sample_rings(fmap, dly, dlx, ell_grid, nphi, order)
    if qu_to_eb:
        t, q, u = rings
        c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
        # the sign that makes E and B those of enmap.map2harm(iau=False)
        rings = np.stack([t, -(q * c2 + u * s2), q * s2 - u * c2])
    a_m = np.full(fmap.shape[:-2] + (mmax + 1, ell_grid.size), np.nan, dtype=complex)
    a_m[..., ok] = _ring_fft(rings, nphi, mmax)
    return a_m


def azimuthal_modes(imap, ell=None, mmax=6, center=None, nphi=None, order=3, qu_to_eb=False):
    """Azimuthal modes a_m(ell) of a flat-sky image.

    Transforms the map, optionally removes the linear phase of an
    off-origin feature, resamples each Fourier ring at uniform psi, and
    Fourier transforms each ring.

    Because the ell spacings come signed from `enmap.lpixshape`,
    psi = atan2(ly, lx) is the azimuth of the map's own frame -- the sky
    frame, +RA toward +dec, for a map in it. This differs from the north
    toward east azimuth used on the sphere (see the Conventions section of
    the module documentation).

    Parameters
    ----------
    imap : (Ny, Nx) or (ncomp, Ny, Nx) real-space enmap. Components are
        decomposed independently unless `qu_to_eb`.
    ell : (nl,) ell values to report; defaults to 256 points spanning
        [0, l_nyquist]. Rings that do not fit entirely on the grid come
        back NaN rather than edge-clamped.
    mmax : highest azimuthal mode returned.
    center : (y0, x0) pixel position put at the Fourier origin, via
        `maps.fourier_recenter`. None leaves the map alone, which is right
        only for something already centred on pixel (0, 0): an off-origin
        feature carries a phase ramp that both forges modes at order
        J_m(ell*d) and destroys the smoothness the ring interpolation
        depends on. Note that the pixel (Ny//2, Nx//2) a centred stamp
        sits on is *not* the Fourier origin -- pass it explicitly to see
        the raw m-content of such a map.
    nphi : azimuthal samples per ring, the same for every ring; default
        max(256, 8*mmax).
    order : spline order for the ring interpolation (scipy's
        `map_coordinates`), default 3.
    qu_to_eb : take a 3-component input as (T, Q, U) and rotate Q/U to
        E/B on each ring before transforming, returning (T, E, B). The
        rotation uses the ring's own psi, where it is exact. Signs follow
        E = -(Q cos2psi + U sin2psi), B = Q sin2psi - U cos2psi, which is
        the E/B of pixell's ``enmap.map2harm`` with its default
        ``iau=False``.

    Returns
    -------
    dict with `ell` and `a_m` of shape (mmax+1, nl), or
    (ncomp, mmax+1, nl) for multi-component input.

    Examples
    --------
    >>> res = harmonic.azimuthal_modes(imap, center=(y0, x0))
    >>> rho, frac = harmonic.mode_metrics(res["a_m"])
    """
    # 2D, or (ncomp, Ny, Nx) with the components independent -- except
    # under qu_to_eb, where exactly three are required and named
    ncomp = 3 if qu_to_eb else (np.shape(imap)[0] if np.ndim(imap) == 3 else None)
    maps.check_enmap(imap, ncomp=ncomp)
    fmap = enmap.fft(imap.astype(np.float64, copy=False), normalize=False)
    if center is not None:
        fmap = maps.fourier_recenter(fmap, center, fourier=True)
    # dividing ring coordinates by a *signed* spacing is what keeps
    # psi = atan2(ly, lx) the map's azimuth rather than a mirrored one on a
    # cdelt1 < 0 map (see `_sample_rings`)
    dly, dlx = enmap.lpixshape(fmap.shape[-2:], fmap.wcs, signed=True)
    if ell is None:
        ell = np.linspace(0.0, _ring_lmax(fmap.shape, dly, dlx), 256)
    if nphi is None:
        nphi = max(256, 8 * mmax)
    if nphi < 2 * mmax + 1:
        raise ValueError(f"nphi={nphi} cannot resolve m up to {mmax}; need >= {2 * mmax + 1}")
    ell = np.asarray(ell, float)
    return dict(ell=ell, a_m=_modes_polar(fmap, dly, dlx, ell, nphi, mmax, order, qu_to_eb))


def _interp_ell(am, ell, lmod):
    """a_m sampled at arbitrary ``|l|``, linearly in ell, zero outside the grid.

    Non-finite entries -- the rings `azimuthal_modes` could not fit on the
    grid -- are dropped rather than interpolated through, since a single
    NaN in the Fourier plane would spread over the whole reconstruction.
    """
    good = np.isfinite(am)
    if not good.any():
        return np.zeros(lmod.shape, complex)
    order = np.argsort(ell[good])
    return np.interp(lmod, ell[good][order], am[good][order], left=0.0, right=0.0)


def azimuthal_image(a_m, ell, shape, wcs, center=None, ms=None):
    """Rebuild a real-space image from its azimuthal modes: the inverse of
    `azimuthal_modes`.

    Evaluates F(ell, psi) = sum_m a_m(ell) exp(i m psi) on the Cartesian
    Fourier grid of (shape, wcs) -- interpolating a_m in ell and reading
    psi off the pixel -- and transforms back.

    Parameters
    ----------
    a_m : (mmax+1, nl) or (ncomp, mmax+1, nl) modes, as returned by
        `azimuthal_modes`.
    ell : (nl,) the grid a_m is sampled on.
    shape, wcs : geometry of the image to build. Its extent may differ
        from that of the map the modes came from, but its pixel area must
        be the same: `azimuthal_modes` uses an unnormalized FFT, so a_m
        scales with the number of pixels per unit area, and a_m does not
        record the pixel size. On pixels of a different area A_out the
        result comes out multiplied by A_out / A_in; multiply by
        A_in / A_out to undo it.
    center : pixel position to put the reconstruction at, undoing the
        `center` given to `azimuthal_modes`. None leaves it on pixel
        (0, 0), i.e. split across the corners.
    ms : which m to include, e.g. `[0]` for the isotropic part alone or
        `[0, 2]` to keep the quadrupole and drop everything else. Default
        is every m present. This is what makes the pair a filter rather
        than only a round trip.

    Returns
    -------
    A real-valued enmap, (Ny, Nx) or (ncomp, Ny, Nx).

    """
    a_m = np.asarray(a_m, dtype=complex)
    ell = np.asarray(ell, dtype=float)
    if a_m.ndim < 2 or a_m.shape[-1] != ell.size:
        raise ValueError(f"a_m {a_m.shape} does not end in len(ell) = {ell.size}")
    ny, nx = shape[-2:]
    mmax = a_m.shape[-2] - 1
    ms = range(mmax + 1) if ms is None else np.atleast_1d(ms)
    if any(m < 0 or m > mmax for m in ms):
        raise ValueError(f"ms must lie in 0..{mmax}, got {list(ms)}")

    # the same fftshifted, signed grid the forward direction samples rings
    # on, so psi here is the psi there
    dly, dlx = enmap.lpixshape((ny, nx), wcs, signed=True)
    ly = ((np.arange(ny) - ny // 2) * dly)[:, None]
    lx = ((np.arange(nx) - nx // 2) * dlx)[None, :]
    lmod, psi = np.hypot(ly, lx), np.arctan2(ly + 0 * lx, lx + 0 * ly)

    flat = a_m.reshape(-1, mmax + 1, ell.size)
    fmap = np.zeros((flat.shape[0], ny, nx), complex)
    for c in range(flat.shape[0]):
        for m in ms:
            am = _interp_ell(flat[c, m], ell, lmod)
            if m == 0:
                fmap[c] += am
            else:
                e = np.exp(1j * m * psi)
                fmap[c] += am * e + (-1) ** m * np.conj(am) * np.conj(e)

    fmap = np.fft.ifftshift(fmap.reshape(a_m.shape[:-2] + (ny, nx)), axes=(-2, -1))
    fmap = enmap.enmap(fmap, wcs)
    if center is not None:
        fmap = maps.fourier_recenter(fmap, -np.asarray(center, dtype=float), fourier=True)
    # `azimuthal_modes` transforms with normalize=False, so this half of the
    # pair carries the 1/npix of the plain unnormalized DFT
    out = np.asarray(enmap.ifft(fmap, normalize=False)).real / (ny * nx)
    return enmap.enmap(out, wcs)


# ---------------------------------------------------------------------------
# metrics on a_m
# ---------------------------------------------------------------------------
def mode_metrics(a_m):
    """Per-ring metrics from a_m of shape (..., mmax+1, nl).

    - rho : ``|a_m| / |a_0|`` (rho[0] == 1 wherever defined)
    - frac : fraction of ring power in each ``|m|`` (+-m pairs counted
      once for m>=1, valid for real-valued real-space maps). Normalized
      over the m actually returned, so it is a budget of the m modes you
      asked for, not of the true total ring power!

    Leading axes are preserved, so a (3, mmax+1, nl) stack comes back as
    (3, mmax+1, nl) with each component normalized by its own monopole.
    """
    amp = np.abs(np.asarray(a_m))
    mono = amp[..., :1, :]  # keepdims, so it broadcasts back over m
    # each m >= 1 stands for the +-m pair, which carry equal power
    w = np.concatenate([[1.0], np.full(amp.shape[-2] - 1, 2.0)])[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = amp / np.where(mono > 0, mono, np.nan)
        p = amp**2 * w
        frac = p / p.sum(axis=-2, keepdims=True)
    return rho, frac


def mode_orientation(a_m, m, deg=False):
    """Position angle of the real-space m-fold pattern, from arg(a_m).

    Returns the axis of the *real-space* structure, in the frame of the
    map the modes came from (+x toward +y, i.e. +RA toward +dec on a
    sky-frame map, but the scan direction on a scan-locked one) and
    modulo its own 2*pi/m symmetry. Every m > 0 angle carries the map's
    frame with it.

    A pattern cos(m(theta - phi)) has an
    order-m Hankel transform carrying (-i)^m = exp(-i m pi/2), so

        -arg(a_m) / m  =  phi + pi/2   (mod 2 pi / m),

    and this undoes it. For m = 4 the shift is a whole period and cancels;
    for m = 2 it is the familiar statement that the Fourier quadrupole
    lies along the real-space *minor* axis, so an ellipse of position
    angle PA has arg(a_2) pointing at PA + 90. Reading a_2's phase as an
    axis without this correction reports the pattern turned 90 degrees.

    The angle is the flat-sky azimuth of this module, not the astronomical
    position angle (north toward east) used by the functions on the sphere:
    on a sky-frame map PA = (90 deg - angle) modulo 360 deg / m. See the
    Conventions section of the module documentation.
    """
    ang = (-np.angle(np.asarray(a_m)[..., m, :]) / m - 0.5 * np.pi) % (2.0 * np.pi / m)
    return np.rad2deg(ang) if deg else ang


def mode_floor(a_m, nhigh=2):
    """Empirical noise floor on rho_m(ell), from the top `nhigh` modes.

    A compact, smooth feature puts essentially nothing in the highest m
    the decomposition returns, so the mean of ``|a_m|`` over those m, divided by
    ``|a_0|``, is what this estimator reports on this map where there is
    nothing to find. A rho_m within a few times the floor is not a
    detection.

    `nhigh` must stay small enough that the modes it averages sit above
    every m being reported, or the test is circular: with mmax=6 it uses
    m = 5, 6, leaving m <= 4 to be judged against it.
    """
    amp = np.abs(np.asarray(a_m))
    with np.errstate(invalid="ignore", divide="ignore"):
        return amp[..., -nhigh:, :].mean(axis=-2) / amp[..., 0, :]


# -------------------------------------------------------------------------
# multipole stacking on the sphere
# -------------------------------------------------------------------------
def catalog_spin_alm(
    ras_deg, decs_deg, lmax, m, weights=None, alphas_deg=None, nthreads=0, epsilon=1e-8
):
    """Spin-m harmonic transform of a delta-function catalog.

    This is the catalog leg of harmonic-space multipole stacking: the
    adjoint spin-m spherical harmonic transform of a sum of weighted
    delta functions,

        {}_m a^c_{l m'} = sum_i w_i e^{i m alpha_i} conj({}_m Y_{l m'}(n_i)),

    computed exactly (no pixelization, no painting kernel) with ducc0's
    general-location adjoint transform. Setting m = 0 recovers the
    ordinary scalar catalog transform used for monopole stacking.

    ducc returns the gradient/curl ("E"/"B") pair of the spin field
    rather than a single complex array. The two are returned separately
    here, already carrying the sign correction that makes

        {}_m a^c = alm_E - i alm_B

    the standard-convention complex coefficient. The correction is +1 for
    m = 0 and -(-1)^m for m >= 1, and has two parts. A unit E coefficient
    at (l, 0) synthesizes, in ducc's spin-s transform, to
    Q(theta) = -(-1)^s sqrt((2l+1)/4pi) d^l_{s0}(theta), where d^l_{s0} is
    the Wigner-d function with the Condon-Shortley phase that
    `harm2profile` resums with. The -1 is the conventional leading minus
    sign of the E/B definition, which ducc's spin-0 transform does not
    have; the (-1)^s is the phase between ducc's spin harmonics and
    d^l_{s0}. The result is checked for m <= 8 against azimuthal moments
    evaluated directly on rings around the source, in
    tests/test_harmonic_sphere.py (test_ring_moments_match_the_estimator),
    and the ducc convention itself in test_ducc_spin_sign_convention.

    Azimuth is measured from local north toward local east, unlike the
    flat-sky functions of this module (see the Conventions section of the
    module documentation).

    Parameters
    ----------
    ras_deg, decs_deg : ndarray
        Source coordinates in degrees.
    lmax : int
        Maximum multipole.
    m : int
        Azimuthal multipole (the spin of the transform); must be >= 0.
    weights : ndarray or None
        Per-source weights; uniform if None.
    alphas_deg : ndarray or None
        Per-source frame rotation angle in degrees. None (the default)
        is the unoriented case, azimuth measured from the local
        meridian. Use the scan/parallactic angle for instrument-frame
        systematics, or a per-object alignment angle for oriented
        stacking.
    nthreads : int
        Threads for the transform; 0 uses all hardware threads.
    epsilon : float
        Requested accuracy of the underlying non-uniform FFT.

    Returns
    -------
    alm_E, alm_B : ndarray
        Complex alm arrays in healpy packing. alm_B is identically zero
        for m = 0.
    """
    if m < 0:
        raise ValueError(
            f"m must be >= 0, got {m}. ducc's transform takes a spin >= 0, and "
            "for a real map the negative-m coefficients add nothing: the "
            "cross-spectrum obeys C^(-m) = conj(C^(m)). Pass abs(m) and "
            "conjugate. Silently returning the +|m| answer here would have "
            "flipped the sign of every sin(m phi) channel."
        )
    ras = np.asarray(ras_deg, dtype=float)
    decs = np.asarray(decs_deg, dtype=float)
    w = np.ones(ras.size) if weights is None else np.asarray(weights, dtype=float)
    loc = np.ascontiguousarray(
        np.column_stack([np.deg2rad(90.0 - decs), np.deg2rad(ras) % (2.0 * np.pi)])
    )
    if alphas_deg is None or m == 0:
        cw, sw = w, np.zeros_like(w)
    else:
        ph = m * np.deg2rad(np.asarray(alphas_deg, dtype=float))
        cw, sw = w * np.cos(ph), w * np.sin(ph)
    vals = np.ascontiguousarray(cw[None, :] if m == 0 else np.stack([cw, sw]))
    alm = ducc0.sht.experimental.adjoint_synthesis_general(
        map=vals,
        loc=loc,
        spin=abs(m),
        lmax=lmax,
        epsilon=epsilon,
        nthreads=nthreads,
    )
    if m == 0:
        return alm[0], np.zeros_like(alm[0])
    sgn = -((-1.0) ** m)
    return sgn * alm[0], sgn * alm[1]


def multipole_cross_spectrum(alm_map, alm_E, alm_B):
    """Complex multipole stack C^(m)_ell from a map and a spin-m catalog.

    Combines the E/B pair returned by catalog_spin_alm into the complex
    cross-spectrum sum_m' a_{l m'} conj({}_m a^c_{l m'}), in pixell's
    alm2cl normalization (i.e. divided by 2l+1). Feeding this to
    harm2profile with the same m gives the real-space azimuthal moment of
    the stack, up to a factor 4 pi.

    The result is complex for m > 0 and both halves are signal: the real
    part is the cos(m phi) channel and the imaginary part -- the alm_B
    term -- the sin(m phi) one, which is the parity channel
    `beam_multipole` describes. Casting this to a real dtype before
    resumming it throws away half the moment, and for the m = 2 of a
    typical map the two halves are comparable in size.

    Parameters
    ----------
    alm_map : ndarray
        Scalar harmonic coefficients of the map.
    alm_E, alm_B : ndarray
        Spin-m catalog coefficients from catalog_spin_alm.

    Returns
    -------
    cl : ndarray
        Complex spectrum starting at ell=0.
    """
    return curvedsky.alm2cl(alm_map, alm_E) - 1j * curvedsky.alm2cl(alm_map, alm_B)


def beam_multipole(cl_m, cl_0, m):
    """Harmonic-space beam multipole b_m/b_0 from the multipole stack.

    `multipole_cross_spectrum` measures the *real-space* azimuthal moment
    of the stack; b_m is its harmonic-space counterpart, and the factor
    i^m is what relates them -- the same i^m that sits between S_m(r) and
    F_m(ell) in the flat-sky pair.

    The result follows the astronomical convention in which anisotropy at
    position angle PA east of north carries a phase exp(-i m PA). So a
    beam elongated *along* the local meridian gives a real, negative
    b_2/b_0, and a coherent northward astrometric offset gives a purely
    imaginary, negative b_1/b_0. The real part measures cos(m phi)
    structure aligned with the meridian and the imaginary part the
    sin(m phi) component rotated by 45/m degrees, which flips sign under
    reflection about the meridian and is therefore a parity channel.

    Any amplitude common to both spectra -- the source fluxes, in a beam
    measurement -- cancels in the ratio.

    Parameters
    ----------
    cl_m : ndarray
        Complex C^(m)_ell, from `multipole_cross_spectrum` (decoupled or
        not, as long as cl_0 had the same treatment). The azimuth it was
        measured in must run from local north towards local east, which
        is what `catalog_spin_alm` does. The flat-sky functions of this
        module use a different azimuth (see the Conventions section of the
        module documentation).
    cl_0 : ndarray
        The m = 0 spectrum, used as the normalization.
    m : int
        Azimuthal multipole.

    Returns
    -------
    bm : ndarray
        Complex b_m/b_0.
    """
    return (1j**m) * np.asarray(cl_m) / np.asarray(cl_0).real


def harm2profile(cl, betas_rad, m=0):
    """Resum a harmonic-space multipole stack into a real-space profile.

    Evaluates S_m(beta) = sum_l (2l+1)/(4 pi) d^l_{m0}(beta) C_l, the
    spin-m generalization of the Legendre transform that turns a beam
    transform into a beam profile. For m = 0 this reproduces
    pixell.utils.beam_transform_to_profile.

    The Wigner-d matrix comes from pywiggle's routine, which is
    general in spin and carries the standard Condon-Shortley phase. An
    equivalent route is a ducc0 spin-m synthesis restricted to mmax = 0,
    whose theta dependence is the same d^l_{m0}; the two agree exactly up
    to ducc's spin sign (see `catalog_spin_alm`).

    Parameters
    ----------
    cl : ndarray
        Spectrum starting at ell=0. May be complex, and for m > 0 usually
        is: the resummation is linear, so a complex C^(m) resums to the
        complex S_m whose imaginary part is the sin(m phi) channel.
    betas_rad : ndarray
        Radii in radians at which to evaluate the profile.
    m : int
        Azimuthal multipole; must be >= 0. Negative m carries no new
        information for a real map, since ``S_-m = conj(S_m)``.

    Returns
    -------
    prof : ndarray
        Profile evaluated at betas_rad, complex if `cl` was.
    """
    if m < 0:
        raise ValueError(
            f"m must be >= 0, got {m}. For a real map S_-m = conj(S_m), so pass "
            "abs(m) and conjugate the result."
        )
    from pywiggle import core as pcore

    # Not dtype=float: a complex cl is the normal case (multipole_cross_spectrum
    # returns one) and casting it would silently discard the sin(m phi) half.
    cl = np.asarray(cl)
    ells = np.arange(cl.size)
    dmat = pcore.compute_wigner_d_matrix(
        cl.size - 1, m, 0, np.cos(np.asarray(betas_rad, dtype=float))
    )
    return dmat @ ((2.0 * ells + 1.0) / (4.0 * np.pi) * cl)


def analytical_tf(modlmap, kfilter, bin_edges):
    """Azimuthally binned transfer function of a flat-sky Fourier-space filter.

    Averages the filter over annuli of constant ``|ell|``. For a filter that is either 0 or 1,
    such as a k-space mask, this is the fraction of modes kept in each annulus and so the
    transfer function of the power spectrum; for a general filter, pass ``kfilter**2`` to get
    the power transfer function. Low-ell annuli hold few modes, so the estimate is noisy there.

    Parameters
    ----------
    modlmap : ndarray
        ``|ell|`` of every Fourier pixel, e.g. ``enmap.modlmap(shape, wcs)``.
    kfilter : ndarray
        The filter, with the shape of ``modlmap``.
    bin_edges : ndarray
        Increasing ``|ell|`` bin edges, as for ``stats.bin2D``.

    Returns
    -------
    centers, tf : ndarray
        Bin centers and the mean of the filter in each bin.
    """
    binner = stats.bin2D(modlmap, bin_edges)
    return binner.bin(np.asarray(kfilter, dtype=float))


# ----------------------------------------------------------------------------
# harmonic coefficient helpers (healpy layout: m major, ell minor, lmax inclusive)


def _alm_ell(lmax):
    """ell of every coefficient in the healpy ordering."""
    return np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])


# ----------------------------------------------------------------------------
# s2wav / S2LET directional wavelet filters, in numpy


def _j_max(L, lam=2.0):
    return int(np.ceil(np.log(L) / np.log(lam)))


def _k_lam(L, lam=2.0, quad_iters=300):
    """Scale-discretised wavelet generating function k_lambda of Leistedt et al. (2013).

    Trapezium integration of exp(-2/(1-s^2))/t with the same rule as s2wav (intervals
    touching the end points 1/lambda and 1 are skipped), so the filters match s2wav.
    """

    def part(a, b):
        if a == b:
            return 0.0
        h = (b - a) / quad_iters
        t = a + h * np.arange(quad_iters + 1)
        keep = ~np.isin(t[:-1], [1 / lam, 1.0]) & ~np.isin(t[1:], [1 / lam, 1.0])
        x = (t - 1.0 / lam) * (2.0 * lam / (lam - 1.0)) - 1.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            f = np.exp(-2.0 / (1.0 - x**2)) / t
        return np.sum(((f[:-1] + f[1:]) * h / 2)[keep])

    J = _j_max(L, lam)
    norm = part(1.0 / lam, 1.0)
    k = np.zeros((J + 2, L))
    for j in range(J + 2):
        for ell in range(L):
            if ell < lam ** (j - 1):
                k[j, ell] = 1
            elif ell <= lam**j:
                k[j, ell] = part(ell / lam**j, 1.0) / norm
    return k


def _directional_filters(L, N, lam=2.0):
    """Directional wavelet filters psi[j, ell, n] for n = 0..N-1, identical to s2wav.

    Returns a real array of shape (J_max+1, L, N). Only n with n + N odd are non-zero
    and psi[l, -n] = psi[l, n]. The scaling function is not needed by the transform.
    """
    if N % 2 == 0:
        raise ValueError("even N gives complex filters; this module supports odd N")
    k = _k_lam(L, lam)
    kappa = np.sqrt(np.clip(k[1:] - k[:-1], 0, None))  # (J+1, L)
    kappa *= np.sqrt((2 * np.arange(L) + 1) / 8.0) / np.pi
    s = np.zeros((L, N))
    for ell in range(1, L):
        gamma = min(N - 1, ell) if (N + ell) % 2 else min(N - 1, ell - 1)
        for n in range(N):
            if (N + n) % 2 and n <= gamma:
                s[ell, n] = np.sqrt(comb(gamma, (gamma - n) // 2) / 2**gamma)
    return np.ascontiguousarray(kappa[:, :, None] * s[None])


def _active_orders(N):
    return [n for n in range(N) if (n + N) % 2 == 1]


def _steering_matrix(N):
    """(2N-1, ncomp) matrix giving W(gamma_g) from (f_0, Re f_n, Im f_n, ...)."""
    gam = 2 * np.pi * np.arange(2 * N - 1) / (2 * N - 1)
    cols = []
    for n in _active_orders(N):
        cols += [np.ones_like(gam)] if n == 0 else [2 * np.cos(n * gam), -2 * np.sin(n * gam)]
    return np.array(cols).T


# ----------------------------------------------------------------------------
# the transform


def _as_mask(mask, shape, wcs):
    """A mask as a 2d enmap. A HEALPix map (ring ordering) is rasterised onto (shape, wcs), an
    array without a wcs must have the shape of that geometry, an enmap keeps its own geometry."""
    if np.ndim(mask) == 1:
        hmask = np.asarray(mask, dtype=np.float64)
        return reproject.healpix2map(hmask, shape, wcs, method="spline", order=0, spin=[0]) > 0.5
    if hasattr(mask, "wcs"):
        mask = enmap.ndmap(np.asarray(mask), mask.wcs)
    elif np.shape(mask) == tuple(shape[-2:]):
        mask = enmap.ndmap(np.asarray(mask), wcs)
    else:
        raise ValueError(
            f"a mask without a wcs must have the shape {tuple(shape[-2:])} of the geometry it "
            f"is on, not {np.shape(mask)}; pass an enmap or a HEALPix map otherwise"
        )
    if mask.ndim != 2 or not wcsutils.is_cyl(mask.wcs):
        raise ValueError("the mask must be a 2d map on a cylindrical geometry")
    return mask


def _resample_mask(mask, shape, wcs):
    """Nearest pixel resampling of a 2d enmap mask onto the cylindrical geometry (shape, wcs).

    Pixels outside the mask's own geometry are False. Both geometries are cylindrical, so
    rows depend only on declination and columns only on RA, and RA wraps around the sky.
    """
    ny, nx = shape[-2:]
    dec = enmap.pix2sky(shape, wcs, [np.arange(ny), np.zeros(ny)])[0]
    ra = enmap.pix2sky(shape, wcs, [np.zeros(nx), np.arange(nx)])[1]
    dec0, ra0 = enmap.pix2sky(mask.shape, mask.wcs, [0, 0])
    y = np.floor(enmap.sky2pix(mask.shape, mask.wcs, [dec, np.full(ny, ra0)])[0] + 0.5)
    x = np.floor(enmap.sky2pix(mask.shape, mask.wcs, [np.full(nx, dec0), ra])[1] + 0.5)
    y, x = y.astype(int), x.astype(int) % utils.nint(360 / abs(mask.wcs.wcs.cdelt[0]))
    oky, okx = (y >= 0) & (y < mask.shape[-2]), x < mask.shape[-1]
    out = np.zeros((ny, nx), bool)
    out[np.ix_(oky, okx)] = np.asarray(mask)[np.ix_(y[oky], x[okx])] > 0.5
    return out


def _jax_available(device_only=False):
    """Whether JAX and a ducc build with the XLA handlers JAX needs where it runs are present.

    JAX on an accelerator needs the handlers of ducc's CUDA backend (``sht_ffi_targets``); on
    a CPU those of the CPU transforms (``sht_ffi_cpu_targets``) serve as well. With
    ``device_only`` this also requires JAX to run on an accelerator, which is what the
    automatic choice of backend asks for.
    """
    try:
        import jax
    except ImportError:
        return False
    on_device = jax.default_backend() != "cpu"
    if device_only and not on_device:
        return False
    names = ["sht_ffi_targets"] + ([] if on_device else ["sht_ffi_cpu_targets"])
    return any(hasattr(ducc0.sht, name) for name in names)


def _default_backend():
    """The backend named by ``SOMA_SCATTERING_BACKEND`` (numpy, jax or auto; default auto).

    ``auto`` picks JAX only where it runs on a GPU. On a CPU the two backends call the same
    ducc transforms, and the numpy one gets there without tracing or compiling.
    """
    b = os.environ.get("SOMA_SCATTERING_BACKEND", "auto").lower()
    if b == "auto":
        return "jax" if _jax_available(device_only=True) else "numpy"
    if b not in ("numpy", "jax"):
        raise ValueError("SOMA_SCATTERING_BACKEND must be numpy, jax or auto")
    return b


def _tick(timer, msg):
    if timer is not None:
        timer.tick(msg)


def _transfer_index(lmax_in, lmax_out):
    """Gather index and validity mask taking healpy ordered alm from lmax_in to lmax_out."""
    ell = _alm_ell(lmax_out)
    m = np.concatenate([np.full(lmax_out + 1 - mm, mm) for mm in range(lmax_out + 1)])
    valid = (ell <= lmax_in) & (m <= lmax_in)
    return np.where(valid, m * (2 * lmax_in + 1 - m) // 2 + ell, 0), valid


class ScatterTransform:
    """Scattering covariances of fields at one band limit, with everything reusable precomputed.

    The base class takes harmonic coefficients (healpy ordering, any maximum multipole, which
    is truncated or zero padded to lmax); ``CARScatterTransform`` and
    ``HealpixScatterTransform`` take maps. The wavelet maps and statistics are always computed
    on full-sky CAR Fejer-1 geometries, one per scale, optionally cut to the rows covering a
    footprint.

    The statistics are returned as the tuple (mean, var, S1, P00, C01, C11) of
    ``s2scat.scatter``, in the same order and normalisation. As in s2scat, ``mean`` is
    ``|f_00| / (2 sqrt(pi))``, the absolute value of the mean, so a field with a negative mean
    is reported with a positive one.

    Parameters
    ----------
    lmax : int
        Maximum multipole of the analysis. The band limit is L = lmax + 1, which is also the
        number of rows of the finest CAR geometry.
    N : int, optional
        Azimuthal band limit of the wavelets, giving 2N-1 orientations. Must be odd. Default 3.
    J_min : int, optional
        Lowest wavelet scale; the highest is ceil(log2 L). Default 2.
    mask : ndarray, enmap or None, optional
        Boolean mask: an enmap on any cylindrical geometry (zero outside it), an array on the
        full-sky Fejer-1 geometry with L rows,
        ``enmap.fullsky_geometry(shape=(L, 2 * L), variant="fejer1")``, or a HEALPix map (ring
        ordering). It is resampled to every scale by nearest pixel. Pixel statistics are
        restricted to it and, with ``cut``, every scale is restricted to the rows covering it.
    cut : bool, optional
        Restrict the per-scale geometries to the rows covering the mask, so that every
        synthesis costs proportionally less (rows are kept whole). Needs a mask. Default False.
    lmax_trunc : bool, optional
        Analyse ``|W_j2|`` only up to the band limit needed by the coarser scales. The analysis
        is the quadrature weighted adjoint on the geometry of the scale, the same on both
        backends. It is not exact for a map that is not band limited, as the modulus of a
        wavelet map never is, so some power from higher multipoles folds into the coefficients
        the second wavelet layer uses. Default True.
    nthreads : int, optional
        Threads for the pixell transforms; 0 (the default) uses all hardware threads. pixell
        gives the environment variable ``OMP_NUM_THREADS``, when it is set, precedence over
        this argument.
    backend : {"numpy", "jax"} or None, optional
        Default from the environment variable ``SOMA_SCATTERING_BACKEND``. The JAX backend
        works in double precision and needs ``jax.config.update("jax_enable_x64", True)`` to
        have been called before the object is created.
    jit : bool, optional
        Compile the JAX transform on first use (JAX backend only). Default True.
    """

    def __init__(
        self,
        lmax,
        N=3,
        J_min=2,
        mask=None,
        cut=False,
        lmax_trunc=True,
        nthreads=0,
        backend=None,
        jit=True,
    ):
        L = lmax + 1
        shape, wcs = enmap.fullsky_geometry(shape=(L, 2 * L), variant="fejer1")
        if mask is not None:
            mask = _as_mask(mask, shape, wcs)
            if not np.any(mask):
                raise ValueError("the mask is empty")
        elif cut:
            raise ValueError("cut restricts the geometries to the rows of a mask, so it needs one")
        self.shape, self.wcs = shape, wcs
        self.backend = backend or _default_backend()
        if self.backend == "jax" and not _jax_available():
            raise RuntimeError(
                "the JAX backend needs JAX and a ducc built with its XLA handlers "
                "(DUCC0_XLA_INCLUDE set to jax.ffi.include_dir(), and DUCC0_USE_CUDA "
                "for a GPU)"
            )
        self.L, self.N, self.J_min, self.J, self.ndir = L, N, J_min, _j_max(L), 2 * N - 1
        self.lmax_trunc, self.nthreads = lmax_trunc, nthreads
        J = self.J
        psi, G, self.orders = _directional_filters(L, N), _steering_matrix(N), _active_orders(N)
        self.ncomp = sum(1 if n == 0 else 2 for n in self.orders)
        cl = np.sqrt(4 * np.pi / (2 * np.arange(L) + 1))
        self.ainfo = curvedsky.alm_info(lmax=L - 1)
        self.Lj = {j: min(int(np.ceil(2.0 ** (j + 1))), L) for j in range(J_min, J + 1)}
        Ls = sorted(set(self.Lj.values()))
        box = None
        if cut:  # declinations of the outer edges of the mask rows
            rows = np.where(np.asarray(mask).any(axis=1))[0]
            box = enmap.pix2sky(mask.shape, mask.wcs, [[rows[0] - 0.5, rows[-1] + 0.5], [0, 0]])[0]
        self.geoms = {}
        for Lj in Ls:
            s, w = enmap.fullsky_geometry(shape=(Lj, 2 * Lj), variant="fejer1")
            # rows covering the declination range, rounded outwards, full columns
            if box is not None:
                y = enmap.sky2pix(s, w, np.array([box, [0, 0]]))[0]
                y0, y1 = (
                    int(np.clip(np.floor(y.min()), 0, s[0])),
                    int(np.clip(np.ceil(y.max()) + 1, 0, s[0])),
                )
                s, w = enmap.subgeo(s, w, pixbox=np.array([[y0, 0], [y1, s[1]]]))
            self.geoms[Lj] = (s, w)
        self.ainfos = {Lj: curvedsky.alm_info(lmax=Lj - 1) for Lj in Ls}
        self.binfos = {
            j2: curvedsky.alm_info(lmax=(self.Lj[j2 - 1] if lmax_trunc else self.Lj[j2]) - 1)
            for j2 in range(J_min + 1, J + 1)
        }
        ell = {Lj: _alm_ell(Lj - 1) for Lj in Ls}
        lmax_ins = {L - 1} | {b.lmax for b in self.binfos.values()}
        self.tidx = {
            (li, Lj): _transfer_index(li, Lj - 1) for li in lmax_ins for Lj in Ls if li != Lj - 1
        }
        fl = {
            (j, n): ((cl * psi[j, :, n])[: self.Lj[j]] * (-1) ** n)[ell[self.Lj[j]]]
            for j in self.Lj
            for n in self.orders
        }
        w, self.norm, self.mask = {}, {}, {}
        for Lj in Ls:
            s, ww = self.geoms[Lj]
            wt = curvedsky.quad_weights(s, ww)[:, None] * np.ones(s)
            self.mask[Lj] = None if mask is None else _resample_mask(mask, s, ww)
            if mask is not None:
                if not self.mask[Lj].any():
                    raise ValueError(
                        f"the mask contains no pixel centre of the band limit {Lj} geometry; "
                        "enlarge the mask or raise J_min"
                    )
                wt *= self.mask[Lj]
            w[Lj], self.norm[Lj] = wt, float(wt.sum())
        # backend specifics: array namespace, constant arrays, transforms and buffers
        self.bufs = {}
        if self.backend == "jax":
            import jax
            import jax.numpy as jnp

            if not jax.config.read("jax_enable_x64"):
                raise ValueError(
                    "the JAX backend works in double precision: call "
                    'jax.config.update("jax_enable_x64", True) before creating the transform'
                )
            self.xp = jnp
            self.G, self.fl, self.w = (
                jnp.asarray(G),
                {k: jnp.asarray(v) for k, v in fl.items()},
                {k: jnp.asarray(v) for k, v in w.items()},
            )
            self.tidx = {k: (jnp.asarray(i), jnp.asarray(v)) for k, (i, v) in self.tidx.items()}
            self.rgeom = {Lj: curvedsky.RingGeometry(*self.geoms[Lj]) for Lj in Ls}
            self._fn = jax.jit(self._stats) if jit else self._stats
        else:
            self.xp, self.G, self.fl, self.w = np, G, fl, w
            self.spinbuf = {
                Lj: np.zeros((2, self.ainfos[Lj].nelem), dtype=np.complex128) for Lj in Ls
            }
            self.comps = {
                Lj: enmap.zeros((self.ncomp,) + self.geoms[Lj][0], self.geoms[Lj][1]) for Lj in Ls
            }
            self._fn = self._stats

    # ---- backend primitives ---------------------------------------------------------------

    def _buf(self, key, shape, dtype=np.float64):
        """A persistent work array for the numpy backend (allocated once per key); zeros for JAX."""
        if self.backend == "jax":
            return self.xp.zeros(shape, dtype)
        if key not in self.bufs or self.bufs[key].shape != tuple(shape):
            self.bufs[key] = np.empty(shape, dtype)
        return self.bufs[key]

    def _set(self, arr, idx, val):
        """arr[idx] = val for numpy, arr.at[idx].set(val) for JAX; returns the updated array."""
        if self.backend == "jax":
            return arr.at[idx].set(val)
        arr[idx] = val
        return arr

    def _synth(self, a, Lj, spin, k):
        """Inverse transform of a (spin 0) or the (E, 0) pair (spin > 0) onto the geometry of Lj.

        Returns the map components; for numpy they are written into rows k.. of the component
        buffer, for JAX a new array is returned.
        """
        if self.backend == "jax":
            # an explicit component axis: (a) for spin 0, the pair (E, B = 0) otherwise
            a = a[None] if spin == 0 else self.xp.stack([a, self.xp.zeros_like(a)])
            return curvedsky.alm2map(
                a, self.rgeom[Lj], spin=[spin], ainfo=self.ainfos[Lj], nthread=self.nthreads
            ).arr
        n = 1 if spin == 0 else 2
        if spin > 0:
            self.spinbuf[Lj][0] = a
            a = self.spinbuf[Lj]
        return curvedsky.alm2map(
            a, self.comps[Lj][k : k + n], spin=[spin], ainfo=self.ainfos[Lj], nthread=self.nthreads
        )

    def _transfer(self, a, lmax_in, Lj):
        """alm at lmax_in truncated or zero padded to the layout of band limit Lj."""
        if lmax_in == Lj - 1:
            return a
        idx, valid = self.tidx[lmax_in, Lj]
        return self.xp.where(valid, a[idx], 0)

    def analysis(self, m, j2):
        """Harmonic coefficients (layout binfos[j2]) of a real map on the geometry of scale j2."""
        wcs = self.geoms[self.Lj[j2]][1]
        m = enmap.devmap(m, wcs) if self.backend == "jax" else enmap.ndmap(m, wcs)
        return curvedsky.map2alm(
            m, ainfo=self.binfos[j2], spin=[0], method="cyl", nthread=self.nthreads
        )

    # ---- the transform -----------------------------------------------------------------------

    def wavelet(self, a, lmax_in, j):
        """Directional wavelet maps (2N-1, ny, nx) of alm a (healpy layout, lmax_in) at scale j."""
        Lj = self.Lj[j]
        a = self._transfer(a, lmax_in, Lj)
        comps, k = [], 0
        for n in self.orders:
            if n == 0:
                comps.append(self._synth(a * self.fl[j, n], Lj, 0, k)[0])
                k += 1
            else:  # E = -c_l psi f, B = 0; f_n = Q - iU
                qu = self._synth(-a * self.fl[j, n], Lj, n, k)
                comps += [qu[0], -qu[1]]
                k += 2
        return self.xp.tensordot(self.G, self.xp.stack(comps), axes=(1, 0))

    def _stats(self, alm, alm2=None, timer=None):
        xp, L, J, J_min, ndir = self.xp, self.L, self.J, self.J_min, self.ndir
        alms = [alm] if alm2 is None else [alm, alm2]
        nf = len(alms)
        means = xp.stack([xp.abs(a[0]) / (2 * xp.sqrt(xp.pi)) for a in alms])
        prod = xp.real(alms[0] * xp.conj(alms[-1]))
        var = (xp.sum(prod[1:L]) + 2 * xp.sum(prod[L:])) / (4 * xp.pi)
        W = [{j: self.wavelet(a, L - 1, j) for j in self.Lj} for a in alms]
        _tick(timer, f"first wavelet layer, {len(self.Lj)} scales")
        S1, P00 = [[] for _ in alms], []
        N2 = [
            {
                j1: self._buf((k, "N2", j1), (ndir, J - j1, ndir) + self.geoms[self.Lj[j1]][0])
                for j1 in range(J_min, J)
            }
            for k in range(nf)
        ]
        for j2 in self.Lj:
            Lj2 = self.Lj[j2]
            w, norm = self.w[Lj2], self.norm[Lj2]
            P00.append(xp.einsum("gtp,gtp,tp->g", W[0][j2], W[-1][j2], w) / norm)
            for k in range(nf):
                absW = xp.abs(W[k][j2])
                S1[k].append(xp.einsum("gtp,tp->g", absW, w) / norm)
                if j2 == J_min:
                    continue
                for g2 in range(ndir):
                    M = self.analysis(absW[g2], j2)
                    for j1 in range(J_min, j2):
                        N2[k][j1] = self._set(
                            N2[k][j1],
                            (slice(None), j2 - j1 - 1, g2),
                            self.wavelet(M, self.binfos[j2].lmax, j1),
                        )
        _tick(timer, "S1, P00 and the second wavelet layer")
        C01, C11 = [], []
        for j1 in range(J_min, J):
            w, norm = self.w[self.Lj[j1]], self.norm[self.Lj[j1]]
            A1, A2, W1 = N2[0][j1], N2[-1][j1], W[0][j1]
            nj2 = A1.shape[1]
            c01, c11 = [], []
            for g1 in range(ndir):
                Ag = A1[g1].reshape(nj2 * ndir, -1)
                Bg = xp.multiply(A2[g1], w).reshape(nj2 * ndir, -1)  # second field, weighted
                c01.append(Bg @ W1[g1].ravel())
                c11.append(Ag @ Bg.T)
            c01 = xp.stack(c01, axis=-1).reshape(nj2, ndir, ndir)  # [j2, gamma2, gamma1]
            c11 = xp.stack(c11).reshape(ndir, nj2, ndir, nj2, ndir).transpose(1, 3, 2, 4, 0)
            C01.append(c01.ravel() * (4 * xp.pi / norm))
            C11.append(c11.ravel() * (4 * xp.pi / norm))
        _tick(timer, "covariances")
        S1 = xp.stack([xp.concatenate(s) for s in S1])
        return (
            means[0] if alm2 is None else means,
            var,
            S1[0] if alm2 is None else S1,
            xp.concatenate(P00),
            xp.concatenate(C01),
            xp.concatenate(C11),
        )

    def _prepare(self, alm):
        # an array of the backend at lmax = L-1; on JAX the transfer is a gather, so that jit,
        # grad and vmap trace through it, and the numpy backend takes JAX arrays to the host
        if self.backend == "numpy":
            alm = np.asarray(alm)
        alm = self.xp.asarray(alm, dtype=self.xp.complex128)
        n = alm.shape[-1]
        lmax_in = (isqrt(8 * n + 1) - 3) // 2
        if (lmax_in + 1) * (lmax_in + 2) // 2 != n:
            raise ValueError(f"{n} coefficients is not the healpy layout of any lmax")
        if lmax_in != self.L - 1 and (lmax_in, self.L) not in self.tidx:
            self.tidx[lmax_in, self.L] = tuple(
                map(self.xp.asarray, _transfer_index(lmax_in, self.L - 1))
            )
        return self._transfer(alm, lmax_in, self.L)

    def __call__(self, alm, alm2=None, timer=None):
        """Scattering statistics of harmonic coefficients.

        Parameters
        ----------
        alm : ndarray
            Harmonic coefficients of a real field in healpy ordering, at any maximum
            multipole. They are truncated or zero padded to the lmax of this object, on either
            backend; JAX arrays stay differentiable through this.
        alm2 : ndarray or None, optional
            A second field, for cross statistics (see ``from_alm``).
        timer : object or None, optional
            An object with a ``tick(msg)`` method, called as each stage finishes. Only used by
            the numpy backend: the JAX backend runs as one compiled function.

        Returns
        -------
        mean, var, S1, P00, C01, C11
            The statistics, as described for the class and for ``from_alm``.
        """
        return self.from_alm(
            self._prepare(alm), None if alm2 is None else self._prepare(alm2), timer
        )

    def from_alm(self, alm, alm2=None, timer=None):
        """Scattering statistics of alm that already have the layout of this object.

        Available on every subclass, so a map based object can also be fed alm directly.
        Unlike ``__call__`` there is no layout transfer, which makes this the function to pass
        to ``jax.vmap`` or ``jax.grad``.

        With a second field ``alm2`` the cross statistics of the pair (f, g) are returned in
        the same flattened layout: ``P00`` = ``<W f . W g>``, ``C01`` = ``<W f . W|W g|>``
        (the first field enters linearly, the second through the modulus), ``C11`` =
        ``<W|W f| . W|W g|>``, and ``var`` the cross variance from the harmonic coefficients;
        ``mean`` and ``S1``, which are single-field quantities, are returned for both fields
        stacked along a leading axis of length 2. With ``alm2`` omitted or identical to ``alm``
        the auto statistics of s2scat are recovered.

        Parameters
        ----------
        alm : ndarray
            Harmonic coefficients of a real field in healpy ordering, with maximum multipole
            exactly lmax = L - 1.
        alm2 : ndarray or None, optional
            A second field with the same layout, for cross statistics.
        timer : object or None, optional
            An object with a ``tick(msg)`` method, called as each stage finishes. It is
            ignored on the JAX backend, which runs as one compiled function.

        Returns
        -------
        mean, var, S1, P00, C01, C11
            The statistics, as described for the class and above.
        """
        if self.backend == "jax":
            return self._fn(alm) if alm2 is None else self._fn(alm, alm2)
        return self._stats(alm, alm2, timer)


class CARScatterTransform(ScatterTransform):
    """Scattering covariances of enmaps on a cylindrical geometry (shape, wcs).

    The geometry may be full-sky or a patch at any resolution. Maps are analysed on it with
    ``curvedsky.map2alm``, which treats the sky outside the geometry as zero, and the pixel
    statistics are restricted to the area the geometry covers (and to ``mask``). For a patch,
    ``cut=True`` also restricts every wavelet scale to the rows covering it, which makes the
    transforms cheaper.

    Parameters
    ----------
    shape, wcs : tuple, astropy.wcs.WCS
        Geometry of the input maps.
    lmax : int or None, optional
        Maximum multipole of the analysis. Default: round(180 deg / pixel height) - 1, which is
        L - 1 for ``enmap.fullsky_geometry(shape=(L, 2 * L))``.
    mask : ndarray, enmap or None, optional
        An enmap on any cylindrical geometry, an array of shape ``shape[-2:]`` on
        (shape, wcs), or a HEALPix map (ring ordering). It is combined with the footprint of
        (shape, wcs).
    niter : int, optional
        Jacobi iterations of ``curvedsky.map2alm``. 0 (the default) is exact on full-sky
        geometries with quadrature weights, such as Fejer-1 and Clenshaw-Curtis; more
        iterations improve the analysis on other full-sky geometries.
    **kwargs
        Other arguments as for ``ScatterTransform``.
    """

    def __init__(self, shape, wcs, lmax=None, mask=None, niter=0, **kwargs):
        self.map_shape, self.map_wcs, self.niter = tuple(shape[-2:]), wcs.deepcopy(), niter
        if lmax is None:
            lmax = utils.nint(180 / abs(wcs.wcs.cdelt[1])) - 1
        footprint = np.ones(self.map_shape, bool)
        if mask is not None:
            footprint = _resample_mask(_as_mask(mask, self.map_shape, wcs), self.map_shape, wcs)
        super().__init__(lmax, mask=enmap.ndmap(footprint, wcs), **kwargs)

    def _to_alm(self, m):
        if not hasattr(m, "wcs"):
            m = (enmap.ndmap if isinstance(m, np.ndarray) else enmap.devmap)(m, self.map_wcs)
        # compared as copies: astropy reports a wcs that was never copied or set up as unequal
        # to one that was, even when the two describe the same geometry
        if tuple(m.shape[-2:]) != self.map_shape or not wcsutils.equal(
            m.wcs.deepcopy(), self.map_wcs, tol=1e-10
        ):
            raise ValueError("the map is not on the geometry this transform was built for")
        return curvedsky.map2alm(m, ainfo=self.ainfo, niter=self.niter, nthread=self.nthreads)

    def __call__(self, imap, imap2=None, timer=None):
        """Return the statistics of a real map on this geometry (cross statistics with imap2).

        The maps are enmaps (or devmaps) on (shape, wcs), or arrays of that shape.
        """
        alms = [self._to_alm(m) for m in [imap, imap2] if m is not None]
        _tick(timer, "map2alm")
        return self.from_alm(*alms, timer=timer)


class HealpixScatterTransform(ScatterTransform):
    """Scattering covariances of HEALPix maps (ring ordering), analysed with ``healpy.map2alm``.

    Parameters
    ----------
    lmax : int
        Maximum multipole of the analysis; the band limit is L = lmax + 1.
    niter : int, optional
        Iterations of ``healpy.map2alm``. 0 (the default) is plain quadrature, accurate to 1e-6
        in the statistics for nside >= L/2; 3 gives 1e-11.
    **kwargs
        Other arguments as for ``ScatterTransform``; ``mask`` may be a HEALPix map.
    """

    def __init__(self, lmax, niter=0, **kwargs):
        super().__init__(lmax, **kwargs)
        self.niter = niter

    def __call__(self, hmap, hmap2=None, timer=None):
        """Return the statistics of a real HEALPix map (cross statistics with hmap2)."""
        import healpy as hp

        alms = [
            hp.map2alm(np.asarray(m, dtype=np.float64), lmax=self.L - 1, iter=self.niter)
            for m in [hmap, hmap2]
            if m is not None
        ]
        _tick(timer, "map2alm")
        return self.from_alm(*alms, timer=timer)
