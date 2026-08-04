"""pytest suite for soma.beams: measured beam transforms vs analytic truth."""

import numpy as np
import pytest
from pixell import enmap, utils

from soma import beams, harmonic, maps

PIX = 0.25  # arcmin
N = 512
FWHM = 1.4  # arcmin
ELL = np.linspace(500.0, 10000.0, 40)

# TAN is what real stamps use; "plain" is the awkward second case -- pixell
# gives it cdelt1 > 0, the opposite handedness, x running *with* +RA. Every
# physical result must be identical on both, since the module reads the sign
# from the WCS instead of assuming it.
PROJECTIONS = ["tan", "plain"]


def relerr(measured, truth, sel=None):
    """Max |measured/truth - 1| over `sel` (default: all finite truth)."""
    measured, truth = np.asarray(measured), np.asarray(truth)
    if sel is None:
        sel = np.isfinite(truth) & (truth != 0)
    return float(np.nanmax(np.abs(measured[sel] - truth[sel]) / np.abs(truth[sel])))


@pytest.fixture(scope="module")
def circular():
    return beams.simulate_beam((N, N), PIX, FWHM)


@pytest.fixture(scope="module")
def elliptical():
    return beams.simulate_beam((N, N), PIX, 1.7, 1.4, angle_deg=30.0)


# ---------------------------------------------------------------------------
# the transfer function itself
# ---------------------------------------------------------------------------
def test_gaussian_transfer_function(circular):
    """b_0(ell)/b_0(0) is exp(-ell^2 sigma^2 / 2) for a Gaussian beam."""
    res = beams.beam_modes(circular, ell=np.concatenate([[0.0], ELL]))
    measured = np.abs(res["b_m"][0]) / np.abs(res["b_m"][0, 0])
    assert relerr(measured[1:], beams.gaussian_bl(ELL, FWHM)) < 1e-6


@pytest.mark.parametrize("fwhm", [0.8, 1.4, 2.5])
def test_gaussian_transfer_function_vs_width(fwhm):
    """The same, across beam sizes -- catches a sigma/FWHM slip."""
    b = beams.simulate_beam((N, N), PIX, fwhm)
    ell = np.linspace(0.0, 3.0 / fwhm * utils.arcmin * utils.fwhm, 30)
    res = beams.beam_modes(b, ell=ell)
    measured = np.abs(res["b_m"][0]) / np.abs(res["b_m"][0, 0])
    assert relerr(measured[1:], beams.gaussian_bl(ell[1:], fwhm)) < 1e-5


def test_circular_beam_has_no_higher_modes(circular):
    """Every m > 0 of a circular beam is zero to the estimator floor."""
    res = beams.beam_modes(circular, ell=ELL, mmax=8)
    for m in range(1, 9):
        assert np.nanmedian(res["rho"][m]) < 1e-8, f"m={m}"


# ---------------------------------------------------------------------------
# ellipses: the Bessel ladder rho_2k = I_k(a) / I_0(a)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2])
def test_elliptical_bessel_ladder(elliptical, k):
    res = beams.beam_modes(elliptical, ell=ELL, mmax=6)
    truth = beams.elliptical_rho(ELL, 1.7, 1.4, k=k)
    assert relerr(res["rho"][2 * k], truth, sel=truth > 1e-4) < 1e-5


def test_elliptical_odd_modes_vanish(elliptical):
    """An ellipse is inversion-symmetric, so all odd m are exactly zero."""
    res = beams.beam_modes(elliptical, ell=ELL, mmax=5)
    for m in (1, 3, 5):
        assert np.nanmedian(res["rho"][m]) < 1e-8, f"m={m}"


def test_fit_recovers_elliptical_gaussian():
    truth = dict(fwhm_arcmin=1.65, fwhm_minor_arcmin=1.35, angle_deg=25.0)
    b = beams.simulate_beam((N, N), PIX, offset_arcmin=(0.4, -0.7), **truth)
    b = enmap.enmap(np.asarray(b) + 1e-3 * np.random.default_rng(0).standard_normal(b.shape), b.wcs)
    f = beams.fit_gaussian(b, elliptical=True)
    assert abs(f["fwhm_major_arcmin"] - truth["fwhm_arcmin"]) < 1e-2
    assert abs(f["fwhm_minor_arcmin"] - truth["fwhm_minor_arcmin"]) < 1e-2
    assert abs(f["angle_deg"] - truth["angle_deg"]) < 0.5
    assert abs(f["dy_arcmin"] - 0.4) < 1e-2
    assert abs(f["dx_arcmin"] + 0.7) < 1e-2


def test_core_only_fit_matches_the_full_one(elliptical):
    """nsigma restricts the fit to the beam core; it must not change it."""
    full = beams.fit_gaussian(elliptical, elliptical=True, nsigma=None)
    core = beams.fit_gaussian(elliptical, elliptical=True)
    for k in ("fwhm_major_arcmin", "fwhm_minor_arcmin", "angle_deg"):
        assert core[k] == pytest.approx(full[k], abs=1e-4), k
    assert np.hypot(*(np.array(core["center_pix"]) - full["center_pix"])) < 1e-4


# ---------------------------------------------------------------------------
# dialled-in multipoles: rho_m = (eps_m/2) (sigma ell / sqrt 2)^m
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,eps", [(2, 0.05), (3, 0.03), (4, 0.02), (5, 0.02)])
def test_multipole_power_law(m, eps):
    b = beams.simulate_beam((N, N), PIX, FWHM, moments={m: eps})
    # about the construction centre: that is what the closed form
    # describes, and an odd moment shifts both the fit and the centroid
    res = beams.beam_modes(b, ell=ELL, mmax=6, center=(N // 2, N // 2))
    truth = beams.multipole_rho(ELL, FWHM, m, eps)
    assert relerr(res["rho"][m], truth, sel=(truth > 1e-4) & (truth < 0.5)) < 1e-4


def test_multipoles_superpose():
    """Injecting several harmonics at once leaves each one unchanged."""
    moments = {2: 0.05, 3: (0.03, 0.7), 4: 0.02}
    b = beams.simulate_beam((N, N), PIX, FWHM, moments=moments)
    res = beams.beam_modes(b, ell=ELL, mmax=6, center=(N // 2, N // 2))
    for m, spec in moments.items():
        eps = spec[0] if np.iterable(spec) else spec
        truth = beams.multipole_rho(ELL, FWHM, m, eps)
        assert relerr(res["rho"][m], truth, sel=(truth > 1e-4) & (truth < 0.5)) < 1e-4


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_multipole_phase_is_recovered(m):
    """`mode_orientation` returns the real-space phase.

    The order-m Hankel transform carries (-i)^m = exp(-i m pi/2), so the
    Fourier pattern is the real-space one turned by a quarter turn. For
    m=4 that is a whole period and invisible; for m=2 it is the familiar
    fact that the Fourier quadrupole lies along the real-space *minor*
    axis. `mode_orientation` undoes it; -arg(b_m)/m would not.
    """
    phase = 0.7
    b = beams.simulate_beam((N, N), PIX, FWHM, moments={m: (0.05, phase)})
    # about the geometric centre, not the fit: an m=1 term displaces the
    # beam, so recentering would absorb it (and flip the residual dipole
    # by pi). Every other m is unaffected by the choice.
    res = beams.beam_modes(b, ell=np.linspace(3000, 9000, 10), mmax=6, center=(N // 2, N // 2))
    got = harmonic.mode_orientation(res["b_m"], m)
    period = 2 * np.pi / m
    assert np.nanmax(np.abs((got - phase + period / 2) % period - period / 2)) < 1e-4


# ---------------------------------------------------------------------------
# translations: the Jacobi-Anger forgery, and its exact removal
# ---------------------------------------------------------------------------
def test_offset_beam_matches_jacobi_anger():
    """Un-recentered, a displaced circular beam shows |J_m(ell d)/J_0(ell d)|."""
    from scipy.special import jv

    off = (0.5, -0.8)
    d = np.hypot(*off) * utils.arcmin
    b = beams.simulate_beam((N, N), PIX, FWHM, offset_arcmin=off)
    ell = np.linspace(300.0, 7000.0, 30)
    b_m = beams.beam_modes(b, ell=ell, center=(N // 2, N // 2))["b_m"]
    for m in (1, 2, 3):
        truth = np.abs(jv(m, ell * d) / jv(0, ell * d))
        assert relerr(np.abs(b_m[m] / b_m[0]), truth) < 1e-4, f"m={m}"


@pytest.mark.parametrize("center", ["fit", "centroid"])
def test_recentering_removes_the_forgery(center):
    b = beams.simulate_beam((N, N), PIX, FWHM, offset_arcmin=(0.5, -0.8))
    res = beams.beam_modes(b, ell=ELL, mmax=4, center=center)
    assert np.nanmedian(res["rho"][1]) < 1e-10
    assert np.nanmedian(res["rho"][2]) < 1e-10


def test_zero_center_is_not_the_forgery():
    """center=(0,0) removes no phase, so on a centered stamp it leaves the
    (-1)^k checkerboard and must NOT reproduce the Bessel curve."""
    from scipy.special import jv

    off = (0.5, -0.8)
    d = np.hypot(*off) * utils.arcmin
    b = beams.simulate_beam((N, N), PIX, FWHM, offset_arcmin=off)
    ell = np.linspace(300.0, 7000.0, 30)
    raw = beams.beam_modes(b, ell=ell, center=(0, 0), rmax_arcmin=1e4)["b_m"]
    truth = np.abs(jv(3, ell * d) / jv(0, ell * d))
    assert np.nanmedian(np.abs(raw[3] / raw[0]) / truth) > 10.0


def test_fitted_center_beats_the_centroid_under_a_bowl():
    """The centroid is the thing center="fit" exists to replace."""
    moments = {2: 0.05, 4: 0.02}
    b = beams.simulate_beam((N, N), PIX, FWHM, moments=moments)
    dy, dx = maps.real_grid(b.shape, b.wcs)
    wid = 0.1 * N * PIX * utils.arcmin
    bowl = 3e-3 * np.exp(-0.5 * (np.hypot(dy - 3 * utils.arcmin, dx) / wid) ** 2)
    bb = enmap.enmap(np.asarray(b) - bowl, b.wcs)
    ell = np.linspace(2000.0, 8000.0, 25)
    cen = beams.beam_modes(bb, ell=ell, center="centroid")
    fit = beams.beam_modes(bb, ell=ell, center="fit")
    for m, eps in moments.items():
        truth = beams.multipole_rho(ell, FWHM, m, eps)
        e_fit, e_cen = relerr(fit["rho"][m], truth), relerr(cen["rho"][m], truth)
        assert e_fit < 1e-2, f"m={m}"
        assert e_fit < 1e-2 * e_cen, f"m={m}: {e_fit:.2e} vs {e_cen:.2e}"


# ---------------------------------------------------------------------------
# polarization: gamma_E, gamma_B from the order-2 Hankel transform
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chi", [0.0, 30.0, 45.0])
def test_leakage_matches_closed_form(chi):
    eps = 0.02
    b = beams.simulate_pol_beam((N, N), PIX, FWHM, eps=eps, chi_deg=chi)
    ell = np.linspace(500.0, 8000.0, 25)
    res = beams.beam_modes(b, ell=ell)
    thE, thB = beams.polarized_leakage_rho(ell, FWHM, eps, chi)
    for got, truth in ((res["gamma_E"].real, thE), (res["gamma_B"].real, thB)):
        if np.max(np.abs(truth)) < 1e-12:  # pure-E or pure-B: exact zero
            assert np.max(np.abs(got)) < 1e-10
        else:
            assert relerr(got, truth) < 1e-5


def test_leakage_is_invariant_under_pointing():
    """gamma_E/B are ratios sharing a common phase, so a shift cancels."""
    kw = dict(eps=0.02, chi_deg=30.0)
    ell = np.linspace(500.0, 8000.0, 20)
    a = beams.beam_modes(beams.simulate_pol_beam((N, N), PIX, FWHM, **kw), ell=ell)
    c = beams.beam_modes(
        beams.simulate_pol_beam((N, N), PIX, FWHM, offset_arcmin=(0.7, -1.1), **kw),
        ell=ell,
    )
    assert relerr(c["gamma_E"].real, a["gamma_E"].real) < 1e-5


def test_polarized_imaginary_parts_are_noise():
    b = beams.simulate_pol_beam((N, N), PIX, FWHM, eps=0.02, chi_deg=20.0)
    res = beams.beam_modes(b, ell=np.linspace(500.0, 8000.0, 20))
    assert np.max(np.abs(res["gamma_E"].imag)) < 1e-15
    assert np.max(np.abs(res["gamma_B"].imag)) < 1e-15


# ---------------------------------------------------------------------------
# nothing depends on the projection or on the WCS handedness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("proj", PROJECTIONS)
def test_projection_handedness_is_read_not_assumed(proj):
    """The sign of cdelt1 must come from the WCS, not from a convention."""
    b = beams.simulate_beam((64, 64), PIX, 2.0, proj=proj)
    dx_pix = enmap.pixshape(b.shape, b.wcs, signed=True)[1]
    assert (dx_pix < 0) == (proj == "tan"), "the two cases must differ in handedness"


@pytest.mark.parametrize("proj", PROJECTIONS)
def test_intensity_results_are_projection_independent(proj):
    """Same beam, either projection: identical FWHMs, PA, offset and modes."""
    kw = dict(
        fwhm_arcmin=1.7,
        fwhm_minor_arcmin=1.4,
        angle_deg=30.0,
        moments={4: 0.02},
        offset_arcmin=(0.4, -0.7),
    )
    ell = np.linspace(1000.0, 9000.0, 20)
    ref = beams.beam_modes(beams.simulate_beam((256, 256), PIX, proj="tan", **kw), ell=ell)
    got = beams.beam_modes(beams.simulate_beam((256, 256), PIX, proj=proj, **kw), ell=ell)
    for k in ("fwhm_major_arcmin", "fwhm_minor_arcmin", "angle_deg", "dx_arcmin"):
        assert got["fit"][k] == pytest.approx(ref["fit"][k], abs=1e-3), k
    for m in (2, 4):
        assert relerr(got["rho"][m], ref["rho"][m]) < 1e-4, f"rho_{m}"
        a = harmonic.mode_orientation(got["b_m"], m, deg=True)
        c = harmonic.mode_orientation(ref["b_m"], m, deg=True)
        assert np.nanmax(np.abs(a - c)) < 1e-2, f"orientation m={m}"


@pytest.mark.parametrize("proj", PROJECTIONS)
def test_leakage_is_projection_independent(proj):
    eps, chi = 0.02, 30.0
    ell = np.linspace(500.0, 8000.0, 20)
    b = beams.simulate_pol_beam((256, 256), PIX, FWHM, eps=eps, chi_deg=chi, proj=proj)
    res = beams.beam_modes(b, ell=ell)
    thE, thB = beams.polarized_leakage_rho(ell, FWHM, eps, chi)
    assert relerr(res["gamma_E"].real, thE) < 1e-5
    assert relerr(res["gamma_B"].real, thB) < 1e-5


@pytest.mark.parametrize("proj", PROJECTIONS)
def test_position_angle_round_trips_in_both_projections(proj):
    """Simulator PA -> fit -> mode_orientation, on both handednesses."""
    for pa in (0.0, 30.0, -55.0, 80.0):
        b = beams.simulate_beam((256, 256), PIX, 2.0, 1.4, angle_deg=pa, proj=proj)
        res = beams.beam_modes(b, ell=np.linspace(2000, 8000, 10), mmax=2)
        assert abs((res["fit"]["angle_deg"] - pa + 90.0) % 180.0 - 90.0) < 0.05
        axis = harmonic.mode_orientation(res["b_m"], 2, deg=True)
        assert np.nanmax(np.abs((axis - pa + 90.0) % 180.0 - 90.0)) < 0.05


# ---------------------------------------------------------------------------
# grid guards and the enmap contract
# ---------------------------------------------------------------------------
def test_rings_off_the_grid_are_nan(circular):
    lnyq = harmonic.l_nyquist(circular.shape, circular.wcs)
    dl = 2.0 * np.pi / (N * PIX * utils.arcmin)
    res = beams.beam_modes(circular, ell=np.array([lnyq, lnyq + dl]))
    assert np.isfinite(res["b_m"][0, 0])
    assert np.isnan(res["b_m"][0, 1])


def test_nphi_below_two_mmax_is_refused(circular):
    with pytest.raises(ValueError):
        beams.beam_modes(circular, ell=ELL, mmax=6, nphi=8)


def test_bad_center_is_refused(circular):
    with pytest.raises(ValueError):
        beams.beam_modes(circular, ell=ELL, center="middle")


@pytest.mark.parametrize("bad", ["array", "wrong_ndim"])
def test_non_enmap_is_refused(circular, bad):
    obj = np.asarray(circular) if bad == "array" else circular[None][None]
    with pytest.raises((TypeError, ValueError)):
        beams.beam_modes(obj)


@pytest.mark.parametrize("dra", [-2.0, 2.0])
def test_center_pix_lands_on_the_peak(dra):
    """center_pix is an array index, dx_arcmin a sky offset.

    On a cdelt1 < 0 map they must disagree in sign, so converting one to
    the other needs the *signed* pixel size.
    """
    b = beams.simulate_beam((128, 128), 0.5, 3.0, offset_arcmin=(1.0, dra))
    cy, cx = beams.fit_gaussian(b)["center_pix"]
    py, px = np.unravel_index(np.argmax(np.asarray(b)), b.shape)
    assert np.hypot(cy - py, cx - px) < 0.5


# ---------------------------------------------------------------------------
# the aperture, and the reported floor
# ---------------------------------------------------------------------------
def test_aperture_is_inert_without_a_bowl(circular):
    ell = np.linspace(2000.0, 9000.0, 15)
    wide = beams.beam_modes(circular, ell=ell, rmax_arcmin=1e3)
    tight = beams.beam_modes(circular, ell=ell, rmax_arcmin=12.0)
    assert relerr(tight["rho"][4], wide["rho"][4]) < 1e-3


def test_aperture_clearing_a_bowl_recovers_the_clean_floor():
    """A hard edge only costs anything where it cuts through signal.

    Circular beam, so any m=4 is the aperture and the bowl talking. An
    edge inside the bowl gets back the bowl-free floor; one cutting
    through it does not.
    """
    ns, ps = 144, 1.0 / 6.0
    clean = beams.simulate_beam((ns, ns), ps, 2.0)
    rr = np.hypot(*maps.real_grid(clean.shape, clean.wcs)) / utils.arcmin
    bowled = enmap.enmap(np.asarray(clean) - 1e-3 * (rr > 7.0), clean.wcs)
    ell = np.linspace(2000.0, 9000.0, 15)

    def m4(arr, rmax):
        res = beams.beam_modes(arr, ell=ell, rmax_arcmin=rmax)
        return float(np.nanmedian(res["rho"][4]))

    floor = m4(clean, None)
    assert m4(bowled, 6.0) == pytest.approx(floor, rel=0.05)  # clears the bowl
    assert m4(bowled, None) > 20 * floor  # cuts through it


def test_reported_floor_sits_below_a_real_signal():
    """The empirical floor must not swallow a mode that is really there."""
    b = beams.simulate_beam((N, N), PIX, FWHM, moments={2: 0.05})
    res = beams.beam_modes(b, ell=np.linspace(2000.0, 8000.0, 20))
    assert np.nanmax(res["floor"] / res["rho"][2]) < 1e-2


# ---------------------------------------------------------------------------
# frames: what a scan-locked T stack alongside sky-frame Q/U does to the
# leakage beams (nothing), and what a careless rotation of Q/U does (plenty)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pa", [0.0, 20.0, 45.0, 90.0])
def test_leakage_is_blind_to_the_T_frame(pa):
    """gamma_E/gamma_B are monopole ratios, so rotating T cannot move them.

    This is the case that matters for real stacks: T coadded in a
    scan-locked frame, Q and U coadded in the sky frame. The relative
    rotation is unknown and irrelevant -- b^T_0 is an azimuthal average.
    """
    n, pix, eps, chi = 256, 0.5, 0.05, 20.0
    ell = np.linspace(500.0, 8000.0, 20)
    qu = beams.simulate_pol_beam((n, n), pix, FWHM, eps=eps, chi_deg=chi)

    def gammas(angle_deg):
        t = beams.simulate_beam((n, n), pix, FWHM, 1.6, angle_deg=angle_deg)
        m = enmap.enmap(np.stack([np.asarray(t), *np.asarray(qu)[1:]]), qu.wcs)
        res = beams.beam_modes(m, ell=ell)
        return res["gamma_E"].real, res["gamma_B"].real

    gE0, gB0 = gammas(0.0)
    gE, gB = gammas(pa)
    assert np.max(np.abs(gE - gE0)) < 1e-9
    assert np.max(np.abs(gB - gB0)) < 1e-9


def test_proper_spin2_rotation_leaves_the_leakage_exactly_alone():
    """Rotating pattern AND polarization basis together changes nothing.

    That pair is what a parallactic-angle coadd does, which is why the
    leakage measured from a sky-frame Q/U stack is the effective leakage
    for the power spectrum rather than an average washed toward zero.
    np.rot90 so the test carries no interpolation error at all.
    """
    n, pix, eps = 256, 0.5, 0.05
    ell = np.linspace(500.0, 8000.0, 20)
    base = beams.simulate_pol_beam((n, n), pix, FWHM, eps=eps, chi_deg=0.0)
    t, q, u = np.asarray(base)
    # alpha = 90 deg: the basis factor exp(2 i alpha) is exactly -1
    rot = enmap.enmap(np.stack([np.rot90(t), -np.rot90(q), -np.rot90(u)]), base.wcs)
    ref, got = beams.beam_modes(base, ell=ell), beams.beam_modes(rot, ell=ell)
    # equal to double precision; not bit-identical only because the FFT of a
    # transposed array accumulates round-off in a different order
    assert np.max(np.abs(got["gamma_E"] - ref["gamma_E"])) < 1e-15
    assert np.max(np.abs(got["gamma_B"] - ref["gamma_B"])) < 1e-15


@pytest.mark.parametrize("alpha", [15.0, 45.0])
def test_spatial_only_rotation_of_QU_turns_E_into_B(alpha):
    """The trap: rotating Q/U without their basis rotates (gE, gB) by 2 alpha.

    Identical to chi -> chi - alpha, so 45 deg of uncompensated rotation
    converts pure T->E leakage into pure T->B. This is what "fixing" a
    scan/sky frame mismatch by rotating the Q/U maps would do.
    """
    n, pix, eps, chi = 256, 0.5, 0.05, 0.0
    ell = np.linspace(500.0, 8000.0, 20)

    def leak(chi_deg):
        b = beams.simulate_pol_beam((n, n), pix, FWHM, eps=eps, chi_deg=chi_deg)
        r = beams.beam_modes(b, ell=ell)
        return r["gamma_E"].real, r["gamma_B"].real

    gE0, gB0 = leak(chi)  # pure E: the reference
    gE, gB = leak(chi - alpha)  # spatially rotated by alpha, basis left behind

    # the magnitude of the leakage vector is untouched...
    assert relerr(np.hypot(gE, gB), np.hypot(gE0, gB0), sel=np.hypot(gE0, gB0) > 1e-6) < 1e-6
    # ...and the vector has turned by exactly -2 alpha
    turn = np.rad2deg(np.arctan2(gB, gE) - np.arctan2(gB0, gE0))
    sel = np.hypot(gE0, gB0) > 1e-6
    assert np.max(np.abs((turn[sel] + 2 * alpha + 180) % 360 - 180)) < 1e-3
    if alpha == 45.0:  # pure E has become pure B
        assert np.max(np.abs(gE)) < 1e-9
        assert relerr(np.abs(gB), np.abs(gE0), sel=sel) < 1e-6


# ---------------------------------------------------------------------------
# handedness, tested without the module's own coordinate convention
# ---------------------------------------------------------------------------
def _sky_beam(shape, wcs, pa_deg=30.0, chi_deg=20.0, eps=0.05):
    """(T, Q, U) of one physical sky beam, built from `enmap.posmap`.

    Deliberately does not use `maps.real_grid`: the simulators share that
    convention with the estimator, so a sign error in it would cancel out
    of every simulate-then-decompose test. Going through posmap -- true
    sky coordinates straight from the WCS -- breaks that circularity, so
    these are the tests that can actually catch a handedness bug.
    """
    pos = enmap.posmap(shape, wcs)
    ddec = np.asarray(pos[0])  # toward +dec
    dra = utils.rewind(np.asarray(pos[1]))  # toward +RA, wrapped to (-pi, pi]
    th = pa_deg * utils.degree
    smaj = 2.0 * utils.arcmin * utils.fwhm
    smin = 1.4 * utils.arcmin * utils.fwhm
    u = np.cos(th) * dra + np.sin(th) * ddec
    v = -np.sin(th) * dra + np.cos(th) * ddec
    t = np.exp(-0.5 * ((u / smaj) ** 2 + (v / smin) ** 2))
    r, phi = np.hypot(ddec, dra), np.arctan2(ddec, dra)
    sig = 1.7 * utils.arcmin * utils.fwhm
    p = eps * (r / (np.sqrt(2.0) * sig)) ** 2 * np.exp(-0.5 * (r / sig) ** 2)
    chi = chi_deg * utils.degree
    q = -p * np.cos(2.0 * phi + 2.0 * chi)
    u_ = -p * np.sin(2.0 * phi + 2.0 * chi)
    return enmap.enmap(np.stack([t, q, u_]), wcs)


def _flip_axis(wcs, axis, n):
    """Mirror a linear WCS along `axis`, describing the same sky."""
    out = wcs.deepcopy()
    out.wcs.cdelt[axis] *= -1.0
    # ra_new(i) = ra_old(n - 1 - i)  <=>  crpix_new + crpix_old = n + 1
    out.wcs.crpix[axis] = n + 1 - wcs.wcs.crpix[axis]
    return out


@pytest.mark.parametrize("flips", [(), (0,), (1,), (0, 1)])
def test_all_four_wcs_handednesses_agree(flips):
    """cdelt1 > 0, cdelt2 < 0, or both: the same sky must give the same answers.

    Includes gamma_B, which is the sharp end of this -- B is parity odd,
    so a mirrored psi would flip its sign while leaving gamma_E alone.
    """
    n, ell = 256, np.linspace(500.0, 8000.0, 25)

    def decompose(*axes):
        shape, wcs = enmap.geometry(pos=[0, 0], res=0.5 * utils.arcmin, shape=(n, n), proj="plain")
        for axis in axes:
            wcs = _flip_axis(wcs, axis, n)
        return beams.beam_modes(_sky_beam(shape, wcs), ell=ell)

    base, got = decompose(), decompose(*flips)
    assert got["fit"]["angle_deg"] == pytest.approx(base["fit"]["angle_deg"], abs=1e-8)
    assert got["fit"]["fwhm_major_arcmin"] == pytest.approx(
        base["fit"]["fwhm_major_arcmin"], abs=1e-10
    )
    # absolute, not relative: rho_4 spans five decades over this ell range, so
    # a relative tolerance would be asking for precision the small end of it
    # never had. These differ at double-precision round-off.
    for m in (2, 4):
        assert np.nanmax(np.abs(got["rho"][0][m] - base["rho"][0][m])) < 1e-14, f"rho_{m}"
    assert np.nanmax(np.abs(got["orient"][0][2] - base["orient"][0][2])) < 1e-8
    assert np.max(np.abs(got["gamma_E"] - base["gamma_E"])) < 1e-15
    assert np.max(np.abs(got["gamma_B"] - base["gamma_B"])) < 1e-15


def test_a_mirrored_map_reports_a_mirrored_offset():
    """What *should* flip does: dx is an offset toward +RA from pixel N//2.

    Mirroring the WCS moves which sky point sits at the centre pixel, so
    the reported dx flips sign while dy and every m-mode stay put. This is
    the counterpart to the test above -- proof it is reading the WCS, not
    ignoring it.
    """
    n = 256
    shape, w0 = enmap.geometry(pos=[0, 0], res=0.5 * utils.arcmin, shape=(n, n), proj="plain")
    w1 = _flip_axis(w0, 0, n)
    a = beams.fit_gaussian(_sky_beam(shape, w0)[0], elliptical=True)
    b = beams.fit_gaussian(_sky_beam(shape, w1)[0], elliptical=True)
    assert a["dx_arcmin"] == pytest.approx(-b["dx_arcmin"], abs=1e-12)
    assert a["dy_arcmin"] == pytest.approx(b["dy_arcmin"], abs=1e-12)
    assert abs(a["dx_arcmin"]) == pytest.approx(0.25, abs=1e-9)  # half a pixel
