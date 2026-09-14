"""
Tests for the inpainting machinery in somapy.maps (and the
correlation functions in somapy.theory): tables against closed forms,
the IQU pixel covariance against Monte-Carlo curvedsky realizations, the
statistical calibration of the conditional fill on band-limited (noise
free) simulations, and the robustness behaviours (RA seam, declination
edges, masked ivar regions, neighbouring holes, seeded randomness,
input immutability).
"""

import numpy as np
import pytest
from pixell import curvedsky, enmap, utils

from somapy import maps as inpaint

ARCMIN = utils.arcmin


def spectra(lmax=200):
    """Smooth, positive-definite TT/EE/BB/TE test spectra up to lmax."""
    ells = np.arange(lmax + 1, dtype=float)
    tt = 1000.0 * np.exp(-((ells / 70.0) ** 2)) + 1.0
    ee = 300.0 * np.exp(-(((ells - 60.0) / 50.0) ** 2)) + 0.5
    bb = 80.0 * np.exp(-(((ells - 100.0) / 60.0) ** 2)) + 0.2
    te = 0.7 * np.sqrt(tt * ee)
    for a in (tt, ee, bb, te):
        a[:2] = 0.0
    return dict(TT=tt, EE=ee, BB=bb, TE=te)


def teb_ps(cl):
    """Pack a spectra dict into the (3, 3, nl) TEB array curvedsky expects."""
    nl = cl["TT"].size
    ps = np.zeros((3, 3, nl))
    ps[0, 0], ps[1, 1], ps[2, 2] = cl["TT"], cl["EE"], cl["BB"]
    ps[0, 1] = ps[1, 0] = cl["TE"]
    return ps


def patch(res_arcmin=8.0, wdeg=4.0, dec0=0.0):
    """A small CAR patch geometry centred at declination dec0."""
    box = np.array([[dec0 - wdeg / 2, -wdeg / 2], [dec0 + wdeg / 2, wdeg / 2]]) * utils.degree
    return enmap.geometry(pos=box, res=res_arcmin * ARCMIN, proj="car")


def test_xi_from_cl_matches_the_pixell_beam_profile():
    # For cl = bl, xi_tt is (up to normalization) the beam image, which
    # pixell computes independently as an exact Legendre sum.
    ells = np.arange(4001, dtype=float)
    fwhm = 10.0 * ARCMIN
    bl = np.exp(-0.5 * ells * (ells + 1) * (fwhm / (8 * np.log(2)) ** 0.5) ** 2)
    r = np.linspace(0.0, 40 * ARCMIN, 200)
    xi = inpaint.xi_from_cl(bl, rmax_rad=r[-1] * 1.01)
    ours = xi["tt"](r)
    ref = utils.beam_transform_to_profile(bl, r)
    assert np.allclose(ours / ours[0], ref / ref[0], atol=2e-3)


def test_iqu_covariance_matches_curvedsky_monte_carlo():
    # The decisive convention test: every block of the analytic pixel
    # covariance must match the empirical covariance of curvedsky
    # realizations with unit scale.
    cl = spectra(120)
    shape, wcs = patch(res_arcmin=40.0, wdeg=8.0, dec0=35.0)
    cy, cx = shape[0] // 2, shape[1] // 2
    pixbox = np.array([[cy - 3, cx - 3], [cy + 3, cx + 3]])
    nsims = 3000
    acc = 0.0
    for i in range(nsims):
        m = curvedsky.rand_map((3,) + shape, wcs, teb_ps(cl), lmax=120, seed=(3, i))
        v = np.asarray(m.extract_pixbox(pixbox), dtype=np.float64).reshape(-1)
        acc = acc + np.outer(v, v)
    emp = acc / nsims

    stamp = enmap.extract_pixbox(enmap.zeros((3,) + shape, wcs), pixbox)
    xi = inpaint.xi_from_cl(cl, rmax_rad=10 * utils.degree, nr=8192)
    ana = inpaint.stamp_cov(stamp.shape, stamp.wcs, xi, ncomp=3)

    n = stamp.shape[-1] * stamp.shape[-2]
    for name, (a, b) in dict(
        TT=(0, 0), TQ=(0, 1), TU=(0, 2), QQ=(1, 1), UU=(2, 2), QU=(1, 2)
    ).items():
        emp_b = emp[a * n : (a + 1) * n, b * n : (b + 1) * n]
        ana_b = ana[a * n : (a + 1) * n, b * n : (b + 1) * n]
        scale = np.sum(emp_b * ana_b) / np.sum(ana_b * ana_b)
        assert scale == pytest.approx(1.0, abs=0.12), f"{name} block scale {scale}"


def test_bandlimited_fill_is_unbiased_and_matches_conditional_variance():
    # Band-limited, noise-FREE simulations: the covariance is near
    # singular, exercising the jitter path, and the fill residuals must
    # be unbiased with variance matching the predicted conditional one.
    lmax = 300
    cl = spectra(lmax)["TT"]
    shape, wcs = patch(res_arcmin=8.0, wdeg=4.0)
    stamp_shape, stamp_wcs = patch(res_arcmin=8.0, wdeg=2.0)
    hole_radius = 20.0 * ARCMIN
    xi = inpaint.xi_from_cl(cl, rmax_rad=6 * utils.degree)
    cov = inpaint.stamp_cov(stamp_shape, stamp_wcs, xi, ncomp=1)
    geo = inpaint.make_geometry(
        stamp_shape, stamp_wcs, hole_radius, cov, ncomp=1, marginalize_mean=False
    )
    cy, cx = shape[0] // 2, shape[1] // 2
    ny, nx = stamp_shape[-2:]
    pixbox = np.array([[cy - ny // 2, cx - nx // 2], [cy + (ny + 1) // 2, cx + (nx + 1) // 2]])

    nsims = 250
    means = []
    sq = []
    for i in range(nsims):
        m = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=(7, i))
        stamp = np.asarray(m.extract_pixbox(pixbox), dtype=np.float64)
        filled = inpaint.inpaint_stamp(stamp, geo)
        resid = (filled - stamp).reshape(-1)[geo["m1"]]
        means.append(resid.mean())
        sq.append(np.mean(resid**2))
    means = np.array(means)
    bias = means.mean() / (means.std() / np.sqrt(nsims))
    assert abs(bias) < 4.0, f"biased fill: {bias:.1f} sigma"
    # In the singular (noise-free) limit the jitter-regularized cond_var is
    # an UPPER bound on the residual variance: the smooth field is nearly
    # perfectly predictable, so demand high accuracy and the bound.
    ratio = np.mean(sq) / np.mean(geo["cond_var"])
    assert ratio < 1.25, f"residual variance ratio {ratio:.2f} exceeds prediction"
    field_var = np.mean(np.asarray(m) ** 2)
    assert np.mean(sq) < 0.05 * field_var, "band-limited fill is not accurate"


def test_noisy_fill_variance_matches_the_prediction():
    # With white noise the covariance is well conditioned and the
    # predicted conditional variance must be exact, not just a bound.
    lmax = 300
    noise = 20.0
    cl = spectra(lmax)["TT"]
    shape, wcs = patch(res_arcmin=8.0, wdeg=4.0)
    stamp_shape, stamp_wcs = patch(res_arcmin=8.0, wdeg=2.0)
    xi = inpaint.xi_from_cl(cl, rmax_rad=6 * utils.degree)
    cov = inpaint.stamp_cov(stamp_shape, stamp_wcs, xi, ncomp=1, noise_uK_arcmin=noise)
    geo = inpaint.make_geometry(
        stamp_shape, stamp_wcs, 20.0 * ARCMIN, cov, ncomp=1, marginalize_mean=False
    )
    cy, cx = shape[0] // 2, shape[1] // 2
    ny, nx = stamp_shape[-2:]
    pixbox = np.array([[cy - ny // 2, cx - nx // 2], [cy + (ny + 1) // 2, cx + (nx + 1) // 2]])
    sigma_pix = np.sqrt(
        (noise * ARCMIN) ** 2 / np.asarray(enmap.pixsizemap(stamp_shape, stamp_wcs))
    )
    rng = np.random.default_rng(99)
    sq = []
    for i in range(250):
        m = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=(8, i))
        stamp = np.asarray(m.extract_pixbox(pixbox), dtype=np.float64)
        stamp = stamp + rng.standard_normal(stamp.shape) * sigma_pix
        filled = inpaint.inpaint_stamp(stamp, geo)
        resid = (filled - stamp).reshape(-1)[geo["m1"]]
        sq.append(np.mean(resid**2))
    ratio = np.mean(sq) / np.mean(geo["cond_var"])
    assert 0.75 < ratio < 1.25, f"residual variance ratio {ratio:.2f}"


def test_planck_style_smooth_map_fill_is_accurate():
    # Harmonically smooth map (30' beam sampled at 8' pixels, no pixel
    # noise): the fill should reproduce the true field to a small
    # fraction of the map rms.
    lmax = 300
    ells = np.arange(lmax + 1, dtype=float)
    bl = np.exp(-0.5 * ells * (ells + 1) * (30 * ARCMIN / (8 * np.log(2)) ** 0.5) ** 2)
    cl = spectra(lmax)["TT"] * bl**2
    shape, wcs = patch(res_arcmin=8.0, wdeg=4.0)
    m = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=11)
    inp = inpaint.Inpainter(cl, hole_arcmin=20.0)
    omap, info = inp(m, ras_deg=[0.0], decs_deg=[0.0])
    assert info["n_filled"] == 1
    resid = np.asarray(omap - m)
    changed = resid != 0
    assert changed.any()
    assert np.sqrt(np.mean(resid[changed] ** 2)) < 0.1 * np.std(np.asarray(m))


def test_point_source_in_the_hole_is_removed():
    lmax = 300
    cl = spectra(lmax)["TT"]
    shape, wcs = patch(res_arcmin=8.0, wdeg=4.0)
    m = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=13).copy()
    modr = np.asarray(enmap.modrmap(shape, wcs))
    src = 5000.0 * np.exp(-0.5 * (modr / (6 * ARCMIN)) ** 2)
    contaminated = m + src
    inp = inpaint.Inpainter(cl, hole_arcmin=25.0, noise_uK_arcmin=5.0)
    omap, info = inp(contaminated, ras_deg=[0.0], decs_deg=[0.0])
    hole = modr < 25.0 * ARCMIN
    assert info["n_filled"] == 1
    # the fill must recover the true underlying field, not the source:
    # measured fill error is ~6 uK mean / ~16 uK max against ~550 uK mean
    # contamination (thresholds carry a ~5x margin)
    err = np.abs(np.asarray(omap - m))[hole]
    before = np.abs(np.asarray(contaminated - m))[hole]
    assert err.mean() < 0.06 * before.mean()
    assert err.max() < 100.0


def test_ra_seam_source_is_filled():
    # A full-RA band with a source at ra ~ 0: the stamp wraps around the
    # seam and must still be cut and written back correctly.
    lmax = 200
    cl = spectra(lmax)["TT"]
    shape, wcs = enmap.band_geometry(
        (-2 * utils.degree, 2 * utils.degree), res=16 * ARCMIN, proj="car"
    )
    m = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=17)
    ra_seam = np.rad2deg(m.pix2sky([0, 0])[1])
    inp = inpaint.Inpainter(cl, hole_arcmin=40.0, noise_uK_arcmin=5.0)
    omap, info = inp(m, ras_deg=[ra_seam], decs_deg=[0.0])
    assert info["n_filled"] == 1
    diff = np.asarray(omap - m)
    ny, nx = shape
    changed_cols = np.flatnonzero((diff != 0).any(axis=0))
    # the hole must straddle the seam: changed pixels on both map edges
    assert changed_cols.min() < nx // 4 and changed_cols.max() > 3 * nx // 4
    assert np.isfinite(np.asarray(omap)).all()


def test_dec_edge_source_uses_reduced_context():
    lmax = 200
    cl = spectra(lmax)["TT"]
    shape, wcs = patch(res_arcmin=8.0, wdeg=4.0)
    m = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=19)
    inp = inpaint.Inpainter(cl, hole_arcmin=20.0, noise_uK_arcmin=5.0)
    omap, info = inp(m, ras_deg=[0.0], decs_deg=[1.9])
    assert info["n_filled"] == 1
    assert info["n_slow"] == 1  # off-map rows excluded from the context
    assert np.isfinite(np.asarray(omap)).all()


def test_zero_ivar_context_pixels_are_ignored():
    # Garbage values in an ivar=0 region must not leak into the fill.
    lmax = 200
    cl = spectra(lmax)["TT"]
    shape, wcs = patch(res_arcmin=8.0, wdeg=4.0)
    m = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=23).copy()
    ivar = enmap.enmap(np.ones(shape), wcs)
    bad = np.zeros(shape, dtype=bool)
    bad[:, : shape[1] // 3] = True
    ivar[bad] = 0.0
    garbled = m.copy()
    garbled[bad] = 1e7

    kw = dict(hole_arcmin=20.0, context_factor=3.0)
    fill_a, _ = inpaint.Inpainter(cl, ivar=ivar, **kw)(garbled, [0.0], [0.0])
    fill_b, _ = inpaint.Inpainter(cl, ivar=ivar, **kw)(m, [0.0], [0.0])
    hole = np.asarray(enmap.modrmap(shape, wcs)) < 20.0 * ARCMIN
    assert np.allclose(np.asarray(fill_a)[hole], np.asarray(fill_b)[hole])


def test_neighbouring_holes_are_excluded_from_the_context():
    # A huge spike at a neighbouring catalog object must not poison this
    # object's fill when mask_others is on (the neighbour's hole pixels
    # are excluded from the context).
    lmax = 200
    cl = spectra(lmax)["TT"]
    shape, wcs = patch(res_arcmin=8.0, wdeg=4.0)
    m = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=29).copy()
    pos = np.asarray(enmap.posmap(shape, wcs))
    spike_mask = np.hypot(pos[0] - np.deg2rad(0.0), pos[1] - np.deg2rad(0.7)) < 10 * ARCMIN
    spiked = m.copy()
    spiked[spike_mask] = 1e6

    ras, decs = [0.0, 0.7], [0.0, 0.0]
    inp = inpaint.Inpainter(cl, hole_arcmin=20.0, noise_uK_arcmin=5.0, mask_others=True)
    filled_spiked, info = inp(spiked, ras, decs)
    inp2 = inpaint.Inpainter(cl, hole_arcmin=20.0, noise_uK_arcmin=5.0, mask_others=True)
    filled_clean, _ = inp2(m, ras, decs)
    hole0 = np.asarray(enmap.modrmap(shape, wcs)) < 20.0 * ARCMIN
    a = np.asarray(filled_spiked)[hole0]
    b = np.asarray(filled_clean)[hole0]
    assert np.abs(a - b).max() < 1.0  # uK; without masking this is ~1e5


def test_add_noise_is_seeded_and_has_the_right_variance():
    lmax = 200
    cl = spectra(lmax)["TT"]
    stamp_shape, stamp_wcs = patch(res_arcmin=8.0, wdeg=2.0)
    xi = inpaint.xi_from_cl(cl, rmax_rad=6 * utils.degree)
    cov = inpaint.stamp_cov(stamp_shape, stamp_wcs, xi, ncomp=1, noise_uK_arcmin=5.0)
    geo = inpaint.make_geometry(stamp_shape, stamp_wcs, 20 * ARCMIN, cov, marginalize_mean=False)
    stamp = np.zeros(stamp_shape)
    draws = np.array(
        [
            inpaint.inpaint_stamp(stamp, geo, add_noise=True, rng=np.random.default_rng(i)).reshape(
                -1
            )[geo["m1"]]
            for i in range(400)
        ]
    )
    ratio = draws.var(axis=0).mean() / geo["cond_var"].mean()
    assert 0.7 < ratio < 1.3
    # determinism under a fixed seed
    a = inpaint.inpaint_stamp(stamp, geo, add_noise=True, rng=np.random.default_rng(5))
    b = inpaint.inpaint_stamp(stamp, geo, add_noise=True, rng=np.random.default_rng(5))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        inpaint.inpaint_stamp(stamp, geo, add_noise=True)


def test_inputs_are_not_mutated_and_context_is_bit_identical():
    lmax = 200
    cl = spectra(lmax)["TT"]
    shape, wcs = patch(res_arcmin=8.0, wdeg=4.0)
    m = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=31)
    ref = m.copy()
    inp = inpaint.Inpainter(cl, hole_arcmin=20.0, noise_uK_arcmin=5.0)
    omap, _ = inp(m, [0.0], [0.0])
    assert np.array_equal(np.asarray(m), np.asarray(ref))
    near_hole = np.asarray(enmap.modrmap(shape, wcs)) < (20.0 + 12.0) * ARCMIN
    diff = np.asarray(omap - m)
    assert (diff[~near_hole] == 0).all()
    assert (diff != 0).any()


def test_geometry_cache_is_reused_across_calls():
    lmax = 200
    cl = spectra(lmax)["TT"]
    shape, wcs = patch(res_arcmin=8.0, wdeg=4.0)
    inp = inpaint.Inpainter(cl, hole_arcmin=20.0, noise_uK_arcmin=5.0)
    ras = [0.0, 1.5, -1.5]  # separations exceed rtot + hole: fast path
    decs = [0.0, 0.1, -0.1]
    m1 = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=37)
    m2 = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=lmax, seed=38)
    _, info1 = inp(m1, ras, decs)
    ngeo = info1["n_geometries"]
    assert ngeo >= 1
    _, info2 = inp(m2, ras, decs)
    assert info2["n_geometries"] == ngeo  # nothing rebuilt on the second sim


def test_iqu_fill_runs_and_is_calibrated():
    cl = spectra(150)
    shape, wcs = patch(res_arcmin=16.0, wdeg=4.0)
    xi = inpaint.xi_from_cl(cl, rmax_rad=6 * utils.degree)
    stamp_shape, stamp_wcs = patch(res_arcmin=16.0, wdeg=2.0)
    noise = 20.0
    cov = inpaint.stamp_cov(stamp_shape, stamp_wcs, xi, ncomp=3, noise_uK_arcmin=noise)
    geo = inpaint.make_geometry(
        stamp_shape, stamp_wcs, 40 * ARCMIN, cov, ncomp=3, marginalize_mean=False
    )
    sigma_pix = np.sqrt(
        (noise * ARCMIN) ** 2 / np.asarray(enmap.pixsizemap(stamp_shape, stamp_wcs))
    )
    rng = np.random.default_rng(123)
    cy, cx = shape[0] // 2, shape[1] // 2
    ny, nx = stamp_shape[-2:]
    pixbox = np.array([[cy - ny // 2, cx - nx // 2], [cy + (ny + 1) // 2, cx + (nx + 1) // 2]])
    nsims = 150
    sq = []
    for i in range(nsims):
        m = curvedsky.rand_map((3,) + shape, wcs, teb_ps(cl), lmax=150, seed=(41, i))
        stamp = np.asarray(m.extract_pixbox(pixbox), dtype=np.float64)
        stamp = stamp + rng.standard_normal(stamp.shape) * sigma_pix * np.array(
            [1.0, np.sqrt(2.0), np.sqrt(2.0)]
        ).reshape(3, 1, 1)
        filled = inpaint.inpaint_stamp(stamp, geo)
        resid = (filled - stamp).reshape(-1)[geo["m1"]]
        sq.append(np.mean(resid**2))
    ratio = np.mean(sq) / np.mean(geo["cond_var"])
    assert 0.7 < ratio < 1.3, f"IQU residual variance ratio {ratio:.2f}"
