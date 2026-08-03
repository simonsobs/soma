"""
Utilities to work with CMB beams, tightly integrated with
pixell.
This includes modeling, simulating, fitting, and
azimuthally decomposing beams.
"""

import numpy as np
import pywiggle
from pixell import curvedsky, enmap, utils
from scipy.optimize import curve_fit, least_squares

from . import harmonic, maps, stats

__all__ = [
    # simulation, and the closed forms to check it against
    "simulate_beam",
    "simulate_pol_beam",
    "gaussian_bl",
    "multipole_rho",
    "elliptical_rho",
    "polarized_leakage_rho",
    # fitting and decomposition
    "fit_gaussian",
    "beam_modes",
    "fit_gaussian_jitter",
    "estimate_jitter_beam",
    # transfer function IO
    "read_beam",
]

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# beam simulation
# ---------------------------------------------------------------------------
def _sim_coords(shape, pix_arcmin, offset_arcmin, proj):
    """(wcs, dy, dx): a stamp geometry and sky offsets from the beam centre."""
    shape, wcs = enmap.geometry(
        pos=[0, 0], res=pix_arcmin * utils.arcmin, shape=shape[-2:], proj=proj
    )
    dy, dx = maps.real_grid(shape, wcs)
    oy, ox = np.asarray(offset_arcmin, float) * utils.arcmin
    return wcs, dy - oy, dx - ox


def simulate_beam(
    shape,
    pix_arcmin,
    fwhm_arcmin,
    fwhm_minor_arcmin=None,
    angle_deg=0.0,
    moments=None,
    amplitude=1.0,
    offset_arcmin=(0.0, 0.0),
    proj="tan",
):
    """A Gaussian beam stamp, optionally elliptical and/or multipolar.

    Circular
        ``simulate_beam(shape, pix, fwhm)``. Transfer function
        exp(-ell^2 sigma^2 / 2), see `gaussian_bl`.

    Elliptical
        add `fwhm_minor_arcmin` and `angle_deg`. On each Fourier ring
        B ~ exp(a cos 2(psi - psi0)) with
        a = ell^2 (smaj^2 - smin^2)/4, so the azimuthal spectrum is the
        Bessel ladder rho_2k = I_k(a)/I_0(a) and all odd m vanish, an
        ellipse being inversion-symmetric. See `elliptical_rho`.

    Dialled-in multipoles
        pass `moments`, which modulates the (circular or elliptical)
        envelope by

            1 + sum_m eps_m (r/(sqrt2 sigma))^m cos(m (theta - phi_m)).

        The radial weight r^m keeps each term smooth at the origin (a
        cos m theta pattern must vanish like r^m there) and makes the
        transform closed-form, giving the pure power law

            ``|b_m|/|b_0| = (eps_m/2) (sigma ell / sqrt2)^m``,

        see `multipole_rho` -- so you can inject a 2% hexadecapole and
        check your pipeline gets it back. Note that closed form assumes a
        *circular* envelope; combining `moments` with an ellipticity is
        allowed but has no simple prediction.

    Parameters
    ----------
    fwhm_minor_arcmin : minor-axis FWHM; defaults to the major, i.e.
        circular.
    angle_deg : major-axis position angle in the sky frame, +RA toward
        +dec -- the convention `fit_gaussian` reports.
    moments : {m: eps} or {m: (eps, phase_rad)}, e.g. {2: 0.05, 3: (0.02,
        0.5)}. Phases are sky-frame. Keep eps small (<~0.3) if the beam
        should stay positive.
    offset_arcmin : (ddec, dra) displacement of the beam from the centre
        pixel, for testing recentering and pointing.
    proj : projection for the stamp geometry, default "tan".

    """
    wcs, dy, dx = _sim_coords(shape, pix_arcmin, offset_arcmin, proj)
    sig = fwhm_arcmin * utils.arcmin * utils.fwhm
    if fwhm_minor_arcmin is None:
        rr = np.hypot(dy, dx) / sig  # circular: radius in units of sigma
    else:
        th = angle_deg * utils.degree
        c, s = np.cos(th), np.sin(th)
        smin = fwhm_minor_arcmin * utils.arcmin * utils.fwhm
        rr = np.hypot((c * dx + s * dy) / sig, (-s * dx + c * dy) / smin)
    beam = np.exp(-0.5 * rr**2)
    if moments:
        # the modulation is defined on the *circular* radius, so that the
        # closed form above holds whatever the envelope is doing
        r, theta = np.hypot(dy, dx), np.arctan2(dy, dx)
        mod = np.ones_like(beam)
        for m, spec in moments.items():
            eps, phase = spec if np.iterable(spec) else (spec, 0.0)
            mod += eps * (r / (np.sqrt(2.0) * sig)) ** m * np.cos(m * (theta - phase))
        beam = beam * mod
    return enmap.enmap(amplitude * beam, wcs)


def simulate_pol_beam(
    shape,
    pix_arcmin,
    fwhm_arcmin,
    eps,
    chi_deg=0.0,
    amplitude=1.0,
    offset_arcmin=(0.0, 0.0),
    proj="tan",
):
    """(3, Ny, Nx) T,Q,U beam stamp with analytically known T->E/B leakage.

    T is a circular Gaussian. The polarized response is the spin-2 pattern

        (Q + iU)(r, theta) = - p(r) exp[i (2 theta + 2 chi)],
        p(r) = amplitude * eps * (r / sqrt(2) sigma)^2 * exp(-r^2/2 sigma^2),

    i.e. a tangential/radial pattern rotated by the mixing angle chi:
    chi = 0 is pure E-leakage, chi = 45 deg pure B. The order-2 Hankel
    transform is closed-form, giving (see `polarized_leakage_rho`)

        gamma_E(ell) = eps (sigma ell / sqrt 2)^2 cos 2chi,
        gamma_B(ell) = eps (sigma ell / sqrt 2)^2 sin 2chi.

    theta and chi are angles in the stamp's frame, from +x toward +y
    (sky frame -- +RA toward +dec -- on a sky-frame stamp).

    """
    wcs, dy, dx = _sim_coords(shape, pix_arcmin, offset_arcmin, proj)
    sig = fwhm_arcmin * utils.arcmin * utils.fwhm
    r = np.hypot(dy, dx)
    th = np.arctan2(dy, dx)
    env = np.exp(-0.5 * (r / sig) ** 2)
    T = amplitude * env
    p = amplitude * eps * (r / (np.sqrt(2.0) * sig)) ** 2 * env
    chi = chi_deg * utils.degree
    Q = -p * np.cos(2.0 * th + 2.0 * chi)
    U = -p * np.sin(2.0 * th + 2.0 * chi)
    return enmap.enmap(np.stack([T, Q, U]), wcs)


# ---------------------------------------------------------------------------
# analytic predictions
# ---------------------------------------------------------------------------
def gaussian_bl(ell, fwhm_arcmin, curved=False):
    """Gaussian beam transfer function, normalized to 1 at ell=0.

    Flat sky (the default) gives exp(-ell^2 sigma^2 / 2); `curved` gives
    the spherical exp(-ell(ell+1) sigma^2 / 2) instead. The two differ by
    one power of ell and matter only at low ell, but a curved-sky beam
    file compared against the flat form picks up a visible ratio there.
    """
    sig = fwhm_arcmin * utils.arcmin * utils.fwhm
    ell = np.asarray(ell, float)
    ll = ell * (ell + 1.0) if curved else ell**2
    return np.exp(-0.5 * ll * sig**2)


def multipole_rho(ell, fwhm_arcmin, m, eps):
    """Analytic ``|b_m|/|b_0|`` for `simulate_beam(..., moments={m: eps})`."""
    sig = fwhm_arcmin * utils.arcmin * utils.fwhm
    return 0.5 * eps * (sig * np.asarray(ell, float) / np.sqrt(2.0)) ** m


def elliptical_rho(ell, fwhm_major_arcmin, fwhm_minor_arcmin, k=1):
    """Analytic ``|b_{2k}|/|b_0|`` for an elliptical Gaussian.

    On a ring, B(ell, psi) = env(ell) * exp(a cos 2(psi - psi0)) with
    a = ell^2 (smaj^2 - smin^2) / 4, hence b_{2k}/b_0 = I_k(a)/I_0(a).
    Odd m vanish identically.
    """
    from scipy.special import iv

    smaj = fwhm_major_arcmin * utils.arcmin * utils.fwhm
    smin = fwhm_minor_arcmin * utils.arcmin * utils.fwhm
    a = 0.25 * np.asarray(ell, float) ** 2 * (smaj**2 - smin**2)
    return np.abs(iv(k, a) / iv(0, a))


def polarized_leakage_rho(ell, fwhm_arcmin, eps, chi_deg=0.0):
    """Analytic (gamma_E, gamma_B) for `simulate_pol_beam`."""
    sig = fwhm_arcmin * utils.arcmin * utils.fwhm
    base = eps * (sig * np.asarray(ell, float) / np.sqrt(2.0)) ** 2
    chi = chi_deg * utils.degree
    return base * np.cos(2.0 * chi), base * np.sin(2.0 * chi)


# ---------------------------------------------------------------------------
# Gaussian fitting
# ---------------------------------------------------------------------------
def fit_gaussian(bmap, elliptical=False, mask=None, p0=None, nsigma=8.0):
    """Least-squares Gaussian fit to a beam enmap.

    Circular model: A exp(-r^2 / 2 sigma^2), params (A, sigma, dy, dx).
    Elliptical model adds a second sigma and a position angle.
    Offsets (dy, dx) are measured from the central pixel (Ny//2, Nx//2).
    An explicit `p0` is in internal units: (A, sigma_rad[, ...], dy_rad,
    dx_rad).

    Returns a dict with user-friendly units:
        amplitude, fwhm_arcmin (or fwhm_major/minor_arcmin, angle_deg),
        dy_arcmin, dx_arcmin, plus 1-sigma errors (``*_err``) from the
        Jacobian, 'success' and 'cost'.

    `nsigma` restricts the least-squares to pixels within that many
    moment-estimated sigmas of the peak (None to use everything). It is
    both a large speedup and more robust, since it keeps any filtering bowl
    out of the fit.
    """
    maps.check_enmap(bmap)
    data = np.asarray(bmap, float)
    dpix = enmap.pixshape(data.shape, bmap.wcs, signed=True)  # (dy, dx), radians
    pix_arcmin = float(np.mean(np.abs(dpix)) / utils.arcmin)
    dy, dx = maps.real_grid(data.shape, bmap.wcs)
    use = np.isfinite(data)
    if mask is not None:
        use &= np.asarray(mask, bool)
    if not use.any():
        raise ValueError("no valid pixels to fit")
    yv, xv, dv = dy[use], dx[use], data[use]

    # moment-based initial guess (robust to where the peak sits); threshold
    # at 5% of the peak so faint noise doesn't dominate the moments
    pos = np.clip(dv - 0.05 * np.nanmax(dv), 0, None)
    tot = pos.sum()
    if tot <= 0:
        raise ValueError("beam has no positive flux to seed the fit")
    y0 = (pos * yv).sum() / tot
    x0 = (pos * xv).sum() / tot
    syy = (pos * (yv - y0) ** 2).sum() / tot
    sxx = (pos * (xv - x0) ** 2).sum() / tot
    sxy = (pos * (yv - y0) * (xv - x0)).sum() / tot
    amp0 = float(dv.max())
    pixrad = pix_arcmin * utils.arcmin

    if nsigma is not None:
        # refit on the core only. The moment sigma above is seeded from
        # pixels over 5% of the peak, so a bowl at 0.1% never enters it
        sig0 = max(np.sqrt(max(0.5 * (sxx + syy), 0.0)), pixrad)
        near = np.hypot(yv - y0, xv - x0) < nsigma * sig0
        if near.sum() > 20:
            yv, xv, dv = yv[near], xv[near], dv[near]

    if not elliptical:
        s0 = max(np.sqrt(max(0.5 * (sxx + syy), 0.0)), pixrad)
        p0 = np.array([amp0, s0, y0, x0]) if p0 is None else np.asarray(p0, float)

        def model(p):
            a, s, cy, cx = p
            return a * np.exp(-0.5 * ((yv - cy) ** 2 + (xv - cx) ** 2) / s**2)

        lb = [-np.inf, 0.1 * pixrad, -np.inf, -np.inf]
        ub = [np.inf] * 4
    else:
        cov = np.array([[syy, sxy], [sxy, sxx]])
        evals, evecs = np.linalg.eigh(cov)  # ascending
        smin0 = max(np.sqrt(max(evals[0], 0.0)), pixrad)
        smaj0 = max(np.sqrt(max(evals[1], 0.0)), smin0)
        vmaj = evecs[:, 1]  # (y, x) components
        th0 = np.arctan2(vmaj[0], vmaj[1])
        p0 = np.array([amp0, smaj0, smin0, th0, y0, x0]) if p0 is None else np.asarray(p0, float)

        def model(p):
            a, smaj, smin, th, cy, cx = p
            c, s = np.cos(th), np.sin(th)
            u = c * (xv - cx) + s * (yv - cy)
            v = -s * (xv - cx) + c * (yv - cy)
            return a * np.exp(-0.5 * ((u / smaj) ** 2 + (v / smin) ** 2))

        lb = [-np.inf, 0.1 * pixrad, 0.1 * pixrad, -np.pi, -np.inf, -np.inf]
        ub = [np.inf, np.inf, np.inf, np.pi, np.inf, np.inf]

    fit = least_squares(lambda p: model(p) - dv, p0, bounds=(lb, ub))

    # 1-sigma parameter errors, by the same route scipy.optimize.curve_fit
    # takes internally: a Moore-Penrose pseudo-inverse of the Jacobian with
    # the null singular values dropped, scaled by the reduced chi-square.
    # Equivalent to inv(J.T @ J) where that is defined, and finite rather
    # than an exception where it is not (a circular beam fitted with the
    # elliptical model leaves the position angle undetermined).
    _, sv, vt = np.linalg.svd(fit.jac, full_matrices=False)
    keep = sv > np.finfo(float).eps * max(fit.jac.shape) * sv[0]
    cov_p = (vt[keep].T / sv[keep] ** 2) @ vt[keep]
    cov_p *= 2.0 * fit.cost / max(dv.size - fit.x.size, 1)
    perr = np.sqrt(np.clip(np.diag(cov_p), 0, None))

    ny, nx = data.shape
    out = {
        "success": bool(fit.success),
        "cost": float(fit.cost),
        # the fit works in signed sky offsets, so converting back to an
        # array index divides by the *signed* pixel size -- with cdelt1 < 0
        # an offset toward +RA is a step to a lower x index
        "center_pix": (
            float(ny // 2 + fit.x[-2] / dpix[0]),
            float(nx // 2 + fit.x[-1] / dpix[1]),
        ),
    }
    if not elliptical:
        a, s, cy, cx = fit.x
        ea, es, ecy, ecx = perr
        out.update(
            amplitude=float(a),
            amplitude_err=float(ea),
            fwhm_arcmin=float(s / utils.fwhm / utils.arcmin),
            fwhm_arcmin_err=float(es / utils.fwhm / utils.arcmin),
            dy_arcmin=float(cy / utils.arcmin),
            dy_arcmin_err=float(ecy / utils.arcmin),
            dx_arcmin=float(cx / utils.arcmin),
            dx_arcmin_err=float(ecx / utils.arcmin),
        )
    else:
        a, smaj, smin, th, cy, cx = fit.x
        if smin > smaj:  # enforce major >= minor
            smaj, smin = smin, smaj
            th += 0.5 * np.pi
            perr[1], perr[2] = perr[2], perr[1]
        th = (th + 0.5 * np.pi) % np.pi - 0.5 * np.pi  # sky PA, mod 180
        ea, esmaj, esmin, eth, ecy, ecx = perr
        out.update(
            amplitude=float(a),
            amplitude_err=float(ea),
            fwhm_major_arcmin=float(smaj / utils.fwhm / utils.arcmin),
            fwhm_major_arcmin_err=float(esmaj / utils.fwhm / utils.arcmin),
            fwhm_minor_arcmin=float(smin / utils.fwhm / utils.arcmin),
            fwhm_minor_arcmin_err=float(esmin / utils.fwhm / utils.arcmin),
            angle_deg=float(np.rad2deg(th)),
            angle_deg_err=float(np.rad2deg(eth)),
            dy_arcmin=float(cy / utils.arcmin),
            dy_arcmin_err=float(ecy / utils.arcmin),
            dx_arcmin=float(cx / utils.arcmin),
            dx_arcmin_err=float(ecx / utils.arcmin),
        )
    return out


# ---------------------------------------------------------------------------
# measured beams
# ---------------------------------------------------------------------------
def _aperture(shape, wcs, center_pix, rmax_arcmin):
    """Zero everything more than `rmax_arcmin` from the beam center.
    """
    ny, nx = shape[-2:]
    if rmax_arcmin is None:  # largest inscribed circle
        pix = np.mean(np.abs(enmap.pixshape(shape[-2:], wcs))) / utils.arcmin
        rmax_arcmin = (min(ny, nx) // 2) * pix
    ref = enmap.pix2sky(shape[-2:], wcs, np.asarray(center_pix, float))
    keep = enmap.modrmap(shape[-2:], wcs, ref=ref) <= rmax_arcmin * utils.arcmin
    if not keep.any():
        raise ValueError("rmax_arcmin excludes every pixel")
    return np.asarray(keep), float(rmax_arcmin)


def beam_modes(bmap, ell=None, mmax=6, center="fit", rmax_arcmin=None, nphi=None, order=3):
    """Azimuthal decomposition of a beam map.

    FFT the stamp, remove a linear phase to put the beam at the origin,
    interpolate the Fourier beam onto uniform-psi rings (cubic spline,
    all rings in one vectorized call), FFT each ring.

    A (3, Ny, Nx) T, Q, U stack is decomposed in the same call: the three
    components share one recentering phase, the flat-sky E/B rotation is
    applied on each uniform-psi ring, and `b_m` comes back as (T, E, B)
    with the leakage beams alongside it.

    Parameters
    ----------
    bmap : (Ny, Nx) or (3, Ny, Nx) = (T, Q, U) real-space pixell enmap.
    ell : (nl,) ell values; defaults to 256 points spanning
        [0, l_nyquist]. Rings beyond the axis full-ring limit return NaN.
    mmax : highest azimuthal mode returned. The top two are what the
        empirical floor is built from, so leave headroom above whatever
        you intend to report (the default 6 reports m <= 4).
    center : the point put at the Fourier origin.

        "fit" (default) takes it from an elliptical Gaussian fit -- the
        right answer for a measured beam, and cheap since `fit_gaussian`
        only fits the core. "centroid" uses the first moment instead,
        which is exact for a clean simulated stamp but not for a real
        one!

        An explicit (y0, x0) in pixels overrides both. Use the geometric
        centre (Ny//2, Nx//2) to see the raw m-content of a stamp as
        constructed. (0, 0) removes no phase at all, which is right only
        for a beam already on pixel (0, 0).

        Whatever centre is used absorbs a true pointing offset and part
        of any intrinsic m=1, which are degenerate.
    rmax_arcmin : aperture radius, beyond which the map is zeroed;
        default is the largest circle inscribed in the stamp. It is
        concentric with `center`, so an explicit centre moves the
        aperture with it -- pass a large rmax if you are giving a centre
        that is not where the beam is.
    nphi : azimuthal samples per ring, the *same for every ring*; default
        max(256, 8*mmax).

    order : spline order for the ring interpolation (scipy's
        `map_coordinates`), default 3.

    Returns
    -------
    dict with

        - ell, b_m : (mmax+1, nl), or (3, mmax+1, nl) in T, E, B order
        - rho, frac : from `mode_metrics`
        - orient : (mmax+1, nl) real-space position angle per mode, in
          degrees, measured in the stamp's frame; see `mode_orientation`
          for the quarter turn it removes
        - floor : empirical noise floor on rho_m -- mean ``|b_m|/|b_0|``
          over the top two m, i.e. what this estimator returns where
          there is nothing to find. A rho_m within a few times the floor
          is not a detection.
        - lmax : largest ell where ``|b_0|`` > 2% of ``|b_0(0)|``; past it rho_m
          is a ratio to almost zero, so starts to be meaningless
        - gamma_E, gamma_B : T->E and T->B fractional leakage beams,
          b^{E,B}_0 / b^T_0, for 3-component input. Complex; for real
          maps the imaginary parts are noise.
        - center : the (y0, x0) actually used, in pixels
        - fit : the `fit_gaussian` dict, when center="fit"
        - rmax_arcmin : the aperture actually used

    Examples
    --------
    >>> res = bt.beam_modes(enmap.read_map("f090_stack.fits"))
    >>> res["rho"][2]           # rho_2 -- (mmax+1, nl) for a 2D stamp
    >>> res["floor"]            # ... and what counts as nothing
    >>> pol = bt.beam_modes(tqu_stack)   # (3, Ny, Nx) in
    >>> pol["rho"][0, 2]        # rho_2 of the T component, (3, mmax+1, nl)
    >>> pol["floor"][0]         # ... against the T floor
    """
    pol = getattr(bmap, "ndim", 0) == 3
    maps.check_enmap(bmap, ncomp=3 if pol else None)
    bmap = bmap.astype(np.float64, copy=False)

    # Center fitting
    fit = None
    if isinstance(center, str):
        if center == "fit":
            fit = fit_gaussian(bmap[0] if pol else bmap, elliptical=True)
            if not fit["success"]:
                raise ValueError(
                    "the Gaussian fit used to center the beam did not "
                    "converge, so its center cannot be trusted; pass an "
                    "explicit center=(y0, x0), or center='centroid' (which "
                    "is wrap-safe) for a clean stamp"
                )
            center = fit["center_pix"]
        elif center == "centroid":
            center = maps.centroid_pixels(bmap[0] if pol else bmap)
        else:
            raise ValueError("center must be 'fit', 'centroid', or (y0, x0)")

    keep, rmax_arcmin = _aperture(bmap.shape, bmap.wcs, center, rmax_arcmin)
    bmap = enmap.samewcs(np.where(keep, bmap, 0.0), bmap)
    bad = ~np.isfinite(bmap)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} non-finite pixel(s) inside the aperture "
            f"(rmax_arcmin={rmax_arcmin:.3g}); Zero them, or shrink rmax_arcmin to "
            "exclude them."
        )

    res = harmonic.azimuthal_modes(
        bmap, ell=ell, mmax=mmax, center=center, nphi=nphi, order=order, qu_to_eb=pol
    )
    ell, b_m = res["ell"], res["a_m"]

    rho, frac = harmonic.mode_metrics(b_m)
    # b_0 of the intensity component: for a (T,E,B) stack that is b_m[0, 0]
    mono = np.abs(b_m[0, 0] if pol else b_m[0])
    # past a b_0 null, rho_m = |b_m|/|b_0| diverges on a beam whose absolute
    # anisotropy is unremarkable, so report where the ratio stops meaning
    # anything rather than letting the caller read the spike as signal
    good = np.isfinite(mono) & (mono > 0.02 * mono[0])
    out = dict(
        ell=ell,
        b_m=b_m,
        rho=rho,
        frac=frac,
        orient=np.stack(
            [
                np.zeros_like(np.abs(b_m[..., 0, :]))
                if m == 0
                else harmonic.mode_orientation(b_m, m, deg=True)
                for m in range(mmax + 1)
            ],
            axis=-2,
        ),
        floor=harmonic.mode_floor(b_m),
        lmax=float(ell[good].max()) if good.any() else float(ell[0]),
        center=tuple(float(v) for v in center),
        fit=fit,
        rmax_arcmin=rmax_arcmin,
    )
    if pol:
        with np.errstate(invalid="ignore", divide="ignore"):
            out.update(gamma_E=b_m[1, 0] / b_m[0, 0], gamma_B=b_m[2, 0] / b_m[0, 0])
    return out


# -------------------------------------------------------------------------
# jitter: the effective/instrument beam ratio
# -------------------------------------------------------------------------
def fit_gaussian_jitter(cents, ratio, err, sel):
    """Jointly fit an amplitude and a Gaussian jitter beam to the ratio.

    The model is A * exp(-ell(ell+1) sigma^2 / 2). Fitting the amplitude
    jointly is important: the overall normalization of the measured
    effective beam is unknown (it depends on the source fluxes), and
    fixing it at low ell would absorb part of the jitter suppression.

    Parameters
    ----------
    cents : ndarray
        Bin centers.
    ratio : ndarray
        Measured jitter beam (effective / instrument beam), arbitrary norm.
    err : ndarray
        1-sigma errors on the ratio.
    sel : ndarray
        Boolean selection of bins used in the fit.

    Returns
    -------
    sigma_arcmin : float
        Best-fit jitter sigma in arcmin (FWHM = sigma * sqrt(8 ln 2)).
    sigma_err_arcmin : float
        1-sigma uncertainty on the jitter sigma in arcmin.
    amp : float
        Best-fit amplitude (the ell=0 normalization of the ratio).
    chi2 : float
        Chi-square of the best fit over the selected bins.
    model : callable
        Function of ell returning the best-fit unit-normalized jitter beam.
    """
    ll1 = cents[sel] * (cents[sel] + 1.0)

    def fmodel(ll1, amp, sigma):
        return amp * np.exp(-0.5 * ll1 * sigma**2)

    popt, pcov = curve_fit(
        fmodel,
        ll1,
        ratio[sel],
        sigma=err[sel],
        absolute_sigma=True,
        p0=[1.0, np.deg2rad(0.5 / 60.0)],
        bounds=([0.0, 0.0], [np.inf, np.deg2rad(30.0 / 60.0)]),
    )
    amp, sigma = popt
    sigma_err = np.sqrt(pcov[1, 1])
    chi2 = np.sum(((ratio[sel] - fmodel(ll1, *popt)) / err[sel]) ** 2)

    def model(ells):
        return np.exp(-0.5 * ells * (ells + 1.0) * sigma**2)

    return np.rad2deg(sigma) * 60.0, np.rad2deg(sigma_err) * 60.0, amp, chi2, model


# -------------------------------------------------------------------------
# transfer function IO
# -------------------------------------------------------------------------
def read_beam(fname, lmax):
    """Read an ACT-style beam transfer function file.

    The file is expected to contain at least two whitespace-separated
    columns: multipole ell and the (possibly unnormalized) harmonic beam
    transform B_ell, as in the beam files released by ACT on LAMBDA.

    Parameters
    ----------
    fname : str
        Path to the beam file.
    lmax : int
        Maximum multipole to return. Must be within the file's support.

    Returns
    -------
    bl : ndarray
        Beam transform of shape (lmax+1,), normalized to 1 at ell=0.
    """
    data = np.loadtxt(fname)
    ells, bl = data[:, 0], data[:, 1]
    if ells[-1] < lmax:
        raise ValueError(f"Beam file {fname} only supports lmax={ells[-1]:.0f} < {lmax}")
    out = np.interp(np.arange(lmax + 1), ells, bl)
    return out / out[0]


# -------------------------------------------------------------------------
# the full jitter beam measurement
# -------------------------------------------------------------------------
def estimate_jitter_beam(
    imap,
    bmask,
    bl,
    ras_deg,
    decs_deg,
    weights=None,
    alphas_deg=None,
    ms=(0, 1, 2, 3, 4),
    lmin=100,
    dell=100,
    apod_deg=1.0,
    deconv_pixwin=True,
    lfit_max=None,
    nthreads=0,
    verbose=True,
    timer=None,
):
    """Estimate the effective beam and its azimuthal multipoles from sources.

    This is the stand-alone core of the pipeline. For each requested
    azimuthal multipole m it forms the harmonic-space multipole stack of
    the map on the catalog,

        C^(m)_ell = sum_m' a^T_{l m'} conj({}_m a^c_{l m'}),

    the cross-spectrum between the map's scalar coefficients and the
    catalog's spin-m coefficients (catalog_spin_alm). Both legs carry the
    same cosine-apodized mask -- the map by multiplication, the catalog
    exactly, by weighting each object by the apodization at its own
    position -- and the pseudo-spectra are mode-decoupled with the
    spin-(m,0) generalization of pywiggle's coupling matrix.

    m = 0 is ordinary monopole stacking and gives the effective beam, to
    whose ratio with the supplied instrument beam an amplitude and a
    Gaussian jitter sigma are jointly fit. m > 0 measures azimuthal
    anisotropy of the stack in the local meridian frame: b_m/b_0 =
    C^(m)/C^(0) with the flux amplitude cancelling. m = 1 responds to a
    coherent astrometric offset, m = 2 to beam ellipticity or scan-locked
    elongation, m = 4 to square-pixel structure. The imaginary ("B")
    channel of each m > 0 is an odd-parity null.

    Parameters
    ----------
    imap : enmap.ndmap
        Intensity map (2d). Assumed to contain the map pixel window
        unless deconv_pixwin is False.
    bmask : enmap.ndmap
        Binary mask (1/0) on the same geometry as imap.
    bl : ndarray
        Instrument beam transform of shape (lmax+1,), starting at ell=0
        and normalized to 1 at ell=0. Its length sets the analysis lmax,
        which must not exceed half the map Nyquist multipole (mask
        spectra are needed to 2 x lmax).
    ras_deg, decs_deg : ndarray
        Source positions in degrees.
    weights : ndarray or None
        Per-source weights (e.g. predicted fluxes); uniform if None.
    alphas_deg : ndarray or None
        Per-source frame rotation in degrees; None is the unoriented
        (local meridian) case. See catalog_spin_alm.
    ms : sequence of int
        Azimuthal multipoles to measure. Must contain 0.
    lmin : int
        Lower edge of the first bandpower bin.
    dell : int
        Bandpower bin width.
    apod_deg : float
        Cosine apodization width in degrees applied to the binary mask.
    deconv_pixwin : bool
        Deconvolve the CAR pixel window from the map before analysis.
    lfit_max : int or None
        Maximum ell used for the preliminary amplitude normalization
        (defaults to lmax/2); the final normalization comes from the
        joint fit and is insensitive to this choice.
    nthreads : int
        Threads for the catalog transforms; 0 uses all available.
    verbose : bool
        Print the in-mask object count and the fit summary. Both values
        are in the returned dict either way.
    timer : Timer or None
        Optional Timer for stage-by-stage progress printing.

    Returns
    -------
    result : dict
        Dictionary with the measurement and reusable intermediates:
        'cents', 'bin_edges' (bandpower binning); 'beff', 'beff_err'
        (normalized effective beam bandpowers); 'ratio', 'ratio_err'
        (jitter beam B_eff/B_instr); 'rsel' (bins used in the fit);
        'sigma_arcmin', 'sigma_err_arcmin', 'fwhm_arcmin', 'chi2', 'dof'
        (joint Gaussian jitter fit); 'jit_model' (callable B_jit(ell));
        'amp' (overall cross-spectrum amplitude in map units); 'cl_td',
        'cl_tt', 'cl_dd', 'err' (decoupled m=0 bandpowers and Knox
        errors); 'bl_b' (instrument beam binned with the theory filter);
        'ms', 'mspec' (per-m dict with 'cl', the complex decoupled
        C^(m)_ell bandpowers, plus 'cl_cc' and 'err' and, for m > 0, the
        complex beam multipole 'bm' = b_m/b_0 and 'bm_err'); 'imap'
        (pixel-window-deconvolved map), 'apod', 'sel', 'wsum', 'alm_T',
        'alm_c0', 'mask_alm', 'wig', 'w2', 'lmax'.
    """

    def tick(msg):
        if timer is not None:
            timer.tick(msg)

    def decouple(wig, pcl, m):
        return wig.decoupled_cl(pcl, "m", spintype=m)["Cls"]

    ms = sorted(set(int(m) for m in ms))
    if 0 not in ms:
        raise ValueError("ms must contain 0 (the monopole sets the normalization)")
    lmax = bl.size - 1
    nyq = int(np.pi / np.abs(np.deg2rad(imap.wcs.wcs.cdelt[0])))
    if 2 * lmax > nyq:
        raise ValueError(f"lmax={lmax} exceeds half the map Nyquist ({nyq // 2})")
    if weights is None:
        weights = np.ones(np.asarray(ras_deg).size)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.mean()
    if lfit_max is None:
        lfit_max = lmax // 2

    if deconv_pixwin:
        imap = enmap.apply_window(imap, pow=-1.0)
        tick("deconvolved map pixel window")
    apod = maps.cosine_apodize(bmask, apod_deg)
    tick(f"apodized mask ({apod_deg} deg cosine)")

    poss = np.array([np.deg2rad(decs_deg), np.deg2rad(ras_deg)])
    sel = bmask.at(poss, order=0) > 0.5
    if verbose:
        print(f"  {sel.sum()}/{sel.size} catalog objects inside the binary mask")
    cw = weights[sel] * apod.at(poss[:, sel], order=1)
    wsum = cw.sum()
    calphas = None if alphas_deg is None else np.asarray(alphas_deg, float)[sel]

    alm_T = curvedsky.map2alm(imap * apod, lmax=lmax)
    mask_alm = curvedsky.map2alm(apod, lmax=2 * lmax)
    apod_alm = curvedsky.transfer_alm(
        curvedsky.alm_info(lmax=2 * lmax), mask_alm, curvedsky.alm_info(lmax=lmax)
    )
    omega = 4.0 * np.pi * maps.wfactor(1, apod)
    tick("computed SHTs of masked map and mask")

    bin_edges = np.arange(lmin, lmax, dell)
    cents = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    wig = pywiggle.Wiggle(lmax, bin_edges=bin_edges, verbose=verbose)
    wig.add_mask("m", mask_alm)
    cl_tt = wig.get_powers(alm_T, alm_T, "m")["TT"]["Cls"]
    w2 = maps.wfactor(2, apod)

    mspec, alm_c0 = {}, None
    for m in ms:
        aE, aB = harmonic.catalog_spin_alm(
            ras_deg[sel],
            decs_deg[sel],
            lmax,
            m,
            weights=cw,
            alphas_deg=calphas,
            nthreads=nthreads,
        )
        aE, aB = aE / wsum, aB / wsum
        if m == 0:
            # remove the mean density so the catalog leg is an overdensity
            aE = aE - apod_alm / omega
            alm_c0 = aE
        pcl = harmonic.multipole_cross_spectrum(alm_T, aE, aB)
        cl = decouple(wig, pcl.real, m) + 1j * decouple(wig, pcl.imag, m)
        clcc = decouple(wig, curvedsky.alm2cl(aE, aE), m)
        mspec[m] = dict(
            cl=cl,
            cl_cc=clcc,
            err=stats.knox_errors(cents, bin_edges, cl_tt, clcc, cl.real, w2),
        )
        tick(f"m={m} catalog transform, cross-spectrum and decoupling")

    cl_td = mspec[0]["cl"].real
    cl_dd, err = mspec[0]["cl_cc"], mspec[0]["err"]
    # b_m/b_0. C^(m) is the real-space azimuthal moment of the stack; i^m
    # conj(.) converts it to the harmonic-space beam multipole in the
    # astronomical convention, in which anisotropy at position angle PA east
    # of north carries the phase exp(-i m PA). 
    for m in ms:
        if m == 0:
            continue
        mspec[m]["bm"] = harmonic.beam_multipole(mspec[m]["cl"], cl_td, m)
        mspec[m]["bm_err"] = mspec[m]["err"] / np.abs(cl_td)

    bl_b = wig.get_theory_filter("m", "m", spintype=0) @ bl
    beff = cl_td.copy()
    beff_err = err.copy()
    fitsel = (cents <= lfit_max) & (bl_b > 0.05)
    amp = np.sum(beff[fitsel] * bl_b[fitsel] / beff_err[fitsel] ** 2) / np.sum(
        bl_b[fitsel] ** 2 / beff_err[fitsel] ** 2
    )
    beff, beff_err = beff / amp, beff_err / np.abs(amp)

    ratio = beff / bl_b
    ratio_err = beff_err / bl_b
    rsel = bl_b > 0.05
    sigma_arcmin, sigma_err_arcmin, ramp, chi2, jit_model = fit_gaussian_jitter(
        cents, ratio, ratio_err, rsel
    )
    amp *= ramp
    beff, beff_err = beff / ramp, beff_err / ramp
    ratio, ratio_err = ratio / ramp, ratio_err / ramp
    fwhm_arcmin = sigma_arcmin * np.sqrt(8.0 * np.log(2.0))
    if verbose:
        print(
            f"Joint amplitude x Gaussian jitter fit: relative amplitude = {ramp:.4f}, "
            f"sigma = {sigma_arcmin:.3f} +- {sigma_err_arcmin:.3f} arcmin "
            f"({sigma_arcmin * 60:.1f} +- {sigma_err_arcmin * 60:.1f} arcsec), "
            f"FWHM = {fwhm_arcmin:.3f} arcmin, chi2/dof = {chi2:.1f}/{rsel.sum() - 2}"
        )
    tick("estimated effective beam and fitted jitter model")

    return dict(
        cents=cents,
        bin_edges=bin_edges,
        beff=beff,
        beff_err=beff_err,
        ratio=ratio,
        ratio_err=ratio_err,
        rsel=rsel,
        sigma_arcmin=sigma_arcmin,
        sigma_err_arcmin=sigma_err_arcmin,
        fwhm_arcmin=fwhm_arcmin,
        chi2=chi2,
        dof=int(rsel.sum()) - 2,
        jit_model=jit_model,
        amp=amp,
        cl_td=cl_td,
        cl_tt=cl_tt,
        cl_dd=cl_dd,
        err=err,
        bl_b=bl_b,
        ms=ms,
        mspec=mspec,
        imap=imap,
        apod=apod,
        sel=sel,
        wsum=wsum,
        alm_T=alm_T,
        alm_c0=alm_c0,
        mask_alm=mask_alm,
        wig=wig,
        w2=w2,
        lmax=lmax,
    )
