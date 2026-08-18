"""
Azimuthal decomposition of flat-sky maps in Fourier space.

The 2D transform of any map can be written on rings of constant ``|ell|`` as

    F(ell, psi) = sum_m a_m(ell) exp(i m psi),   psi = atan2(ly, lx)

    a_m(ell) = < F(ell, psi) exp(-i m psi) >_ring

with a_0 the isotropic part, a_2 the quadrupole, a_4 the hexadecapole and
so on. `azimuthal_modes` measures those coefficients for a generic image;
`mode_metrics`, `mode_orientation` and `mode_floor` interpret them.

Each ring is resampled at uniform psi and Fourier transformed, rather
than averaged over the grid pixels that happen to fall in an annulus.
That distinction is the reason this module exists. A square lattice
samples azimuth with exactly the four-fold symmetry of the grid, so
annulus averaging mixes m with m +- 4 and puts a spurious m=4 (and
m=8, ...) at the 1e-3..1e-2 level under any map, at *any* pixel size --
finer pixels only extend the grid to higher ell, leaving the lattice at
fixed ell unchanged. Uniform sampling in psi restores discrete
orthogonality and drops a circularly symmetric map's spurious m=4 to
~1e-10.

On the sphere the same decomposition is done about a *catalog* of
positions rather than about the origin of one image, and it becomes a
cross-spectrum: `catalog_spin_alm` transforms the catalog on a spin-m
harmonic basis, `multipole_cross_spectrum` crosses that with the map, and
`harm2profile` resums the result back to the real-space azimuthal moment.
`beam_multipole` turns the cross-spectrum into b_m/b_0. Those four are the
curved-sky half of this module, and share nothing with the flat-sky half
except the idea.
"""

import ducc0
import numpy as np
from pixell import curvedsky, enmap
from pywiggle import _wiggle
from scipy.ndimage import map_coordinates
from scipy.integrate import trapezoid

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
        rings = np.stack([t, q * c2 + u * s2, -q * s2 + u * c2])
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
    frame, +RA toward +dec, for a map in it.

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
        E = Q cos2psi + U sin2psi, B = -Q sin2psi + U cos2psi.

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
    shape, wcs : geometry of the image to build. It need not be the one
        the modes came from; only the Fourier grid it implies is used.
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


def integrated_multipole_fractions(frac, ell, a_m):
    """
    Compute a single scalar fraction of total power per multipole m 
    by power-weighting and integrating the per-ring fractions across ell.
    
    Parameters
    ----------
    frac : ndarray
        Per-ring power fractions from `mode_metrics()`.
        Has shape (mmax + 1, nl).
    ell : ndarray
        Multipole values corresponding to the columns of frac.
        Has shape (nl,)
    a_m : ndarray
        Azimuthal mode coefficients used to weight rings by monopole power.
        Has shape (mmax + 1, nl).
        
    Returns
    -------
    scalar_fractions : ndarray
        Array containing the integrated power fraction for each multipole m.
        Has shape (mmax + 1,)
    """
    mmax = frac.shape[0] - 1
    ring_weights = (np.abs(a_m)[0] ** 2) * ell
    valid = np.isfinite(frac).all(axis=0) * np.isfinite(ell) * (ring_weights > 0)
    if not valid.any():
        return np.zeros(mmax + 1, dtype=float)
        
    scalar_fractions = np.empty(mmax + 1, dtype=float)
    denominator = trapezoid(ring_weights[valid], ell[valid])
    
    for m in range(mmax + 1):
        if denominator > 0:
            numerator = trapezoid(frac[m, valid] * ring_weights[valid], ell[valid])
            scalar_fractions[m] = float(numerator / denominator)
        else:
            scalar_fractions[m] = 0.0
            
    return scalar_fractions


def integrated_multipole_angles(a_m, ell, deg):
    """
    Compute a single power-weighted orientation angle (in degrees) for each 
    multipole across ell rings.
    
    Parameters
    ----------
    a_m : ndarray
        Azimuthal mode coefficients.
        Has shape (mmax + 1, nl).
    ell : ndarray
        Multipole values corresponding to the columns of a_m.
        Has shape (nl,)
    deg : bool
       If True return angles in degrees.
        
    Returns
    -------
    scalar_angles : ndarray of shape (mmax + 1,)
        Array containing the integrated orientation angle in degrees for each m 
        Index 0 is 0.0 since monopole has no orientation.
    """
    mmax = a_m.shape[0] - 1
    ring_weights = (np.abs(a_m)[0] ** 2) * ell
    valid = np.isfinite(ell) & (ring_weights > 0)
    scalar_angles = np.zeros(mmax + 1, dtype=float)
    if not valid.any():
        return scalar_angles
        
    for m in range(1, mmax + 1):
        ring_angs = mode_orientation(a_m, m, deg=False)
        v_idx = valid * np.isfinite(ring_angs)
        if not v_idx.any():
            continue
        valid_angs = ring_angs[v_idx]
        valid_w = ring_weights[v_idx]
        complex_mean = np.sum(valid_w * np.exp(1j * m * valid_angs)) / np.sum(valid_w)
        scalar_angles[m] = (np.angle(complex_mean) / m) % (2 * np.pi / m)

    if deg:
        scalar_angles = np.rad2deg(scalar_angles)
        
    return scalar_angles


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

    the standard-convention complex coefficient: ducc's spin>0
    transforms carry the conventional leading minus sign that its spin-0
    transform does not, so the correction is +1 for m = 0 and -(-1)^m for
    m >= 1. That is verified to machine precision for m <= 8 by
    test_multipoles.py, against azimuthal moments evaluated directly on
    rings around the source.

    Parameters
    ----------
    ras_deg, decs_deg : ndarray
        Source coordinates in degrees.
    lmax : int
        Maximum multipole.
    m : int
        Azimuthal multipole (the spin of the transform).
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
        is what `catalog_spin_alm` does.
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

    The Wigner-d matrix comes from pywiggle's compiled routine, which is
    general in spin and carries the standard Condon-Shortley phase. An
    equivalent route is a ducc0 spin-m synthesis restricted to mmax = 0,
    whose theta dependence is the same d^l_{m0}; the two agree exactly up
    to ducc's spin sign (see `catalog_spin_alm`).

    Parameters
    ----------
    cl : ndarray
        Spectrum starting at ell=0.
    betas_rad : ndarray
        Radii in radians at which to evaluate the profile.
    m : int
        Azimuthal multipole.

    Returns
    -------
    prof : ndarray
        Profile evaluated at betas_rad.
    """
    cl = np.asarray(cl, dtype=float)
    ells = np.arange(cl.size)
    dmat = _wiggle._compute_wigner_d_matrix(
        cl.size - 1, abs(m), 0, np.cos(np.asarray(betas_rad, dtype=float))
    )
    return dmat @ ((2.0 * ells + 1.0) / (4.0 * np.pi) * cl)


def analytical_tf(modlmap, kfilter, bin_edges):
    """
    Simple analytic filter for k-space masking.
    Inaccurate at low ell.
    """
    binner = stats.bin2D(modlmap, bin_edges)
    return binner.bin(np.asarray(kfilter, dtype=float))
