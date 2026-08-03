"""pytest suite for soma.beams.estimate_jitter_beam.
"""

import numpy as np
import pytest
from pixell import curvedsky, enmap, utils

from soma import beams, harmonic, maps

RES = 8.0  # arcmin
FWHM = 20.0  # arcmin instrument beam, well sampled by RES
JITTER = 15.0  # arcmin of extra Gaussian smearing to recover
RATIO = 1.4  # injected axis ratio
PA = 30.0  # injected position angle, east of north
OFFSET = 120.0  # arcsec, injected coherent northward offset
NSRC = 4000


def _catalog(seed):
    rng = np.random.default_rng(seed)
    lo, hi = np.sin(np.deg2rad(-50.0)), np.sin(np.deg2rad(20.0))
    decs = np.rad2deg(np.arcsin(rng.uniform(lo, hi, NSRC)))
    return rng, rng.uniform(0.0, 360.0, NSRC), decs, rng.lognormal(0.0, 0.7, NSRC)


def build(ratio, pa_deg, offset_arcsec, seed=4):
    """A masked full-sky map of sources with a known effective beam.

    An isotropic, unoffset beam is built in *harmonic* space -- the exact
    spin-0 catalog transform times the analytic b_ell, synthesized once --
    so the effective beam is exactly the Gaussian asked for, with no
    painting or pixel-sampling error. That is what lets the m=0 test
    assert the injected jitter to a couple of per cent. Anisotropy has no
    such closed form on this side, so those maps are painted, and the m>0
    tests ask only about phases and signs, which a per-cent amplitude
    error does not move.
    """
    rng, ras, decs, fluxes = _catalog(seed)
    shape, wcs = enmap.fullsky_geometry(res=RES * utils.arcmin)
    lmax = int(np.pi / (RES * utils.arcmin)) // 2
    fwhm_eff = np.hypot(FWHM, JITTER)

    if ratio == 1.0 and offset_arcsec == 0.0:
        alm = harmonic.catalog_spin_alm(ras, decs, lmax, 0, weights=fluxes * 1e3)[0]
        bl_eff = beams.gaussian_bl(np.arange(lmax + 1), fwhm_eff, curved=True)
        imap = curvedsky.alm2map(curvedsky.almxfl(alm, bl_eff), enmap.zeros(shape, wcs))
    else:
        # split the FWHM at fixed geometric mean, so the solid angle is
        # untouched by `ratio`
        q = np.sqrt(ratio)
        imap = maps.paint_elliptical(
            shape,
            wcs,
            ras,
            decs + offset_arcsec / 3600.0,
            fluxes * 1e3,
            fwhm_eff * q,
            fwhm_eff / q,
            pa_deg,
        )
    imap += rng.standard_normal(imap.shape).astype(np.float32) * 0.5

    # a declination strip, so the mask is not trivial
    mdecs = np.rad2deg(enmap.pix2sky(shape, wcs, [np.arange(shape[0]), np.zeros(shape[0])])[0])
    strip = ((mdecs > -55.0) & (mdecs < 25.0)).astype(np.float32)
    mask = enmap.enmap(np.repeat(strip[:, None], shape[1], axis=1), wcs)

    return dict(
        imap=imap,
        mask=mask,
        bl=beams.gaussian_bl(np.arange(lmax + 1), FWHM, curved=True),
        ras=ras,
        decs=decs,
        fluxes=fluxes,
        lmax=lmax,
    )


def run(sim, **kw):
    kw = dict(weights=sim["fluxes"], lmin=100, dell=50, deconv_pixwin=False, verbose=False) | kw
    return beams.estimate_jitter_beam(
        sim["imap"], sim["mask"], sim["bl"], sim["ras"], sim["decs"], **kw
    )


@pytest.fixture(scope="module")
def isotropic():
    return build(ratio=1.0, pa_deg=0.0, offset_arcsec=0.0)


@pytest.fixture(scope="module")
def anisotropic():
    return build(ratio=RATIO, pa_deg=PA, offset_arcsec=OFFSET)


@pytest.fixture(scope="module")
def result(anisotropic):
    return run(anisotropic, ms=(0, 1, 2))


def test_recovers_the_injected_jitter_width(isotropic):
    """m=0 must return the Gaussian that was convolved in."""
    res = run(isotropic, ms=(0,))
    assert res["dof"] > 3
    fwhm = res["sigma_arcmin"] * np.sqrt(8.0 * np.log(2.0))
    assert fwhm == pytest.approx(JITTER, rel=0.03)
    assert res["sigma_err_arcmin"] > 0


def test_an_isotropic_beam_has_no_higher_multipoles(isotropic):
    res = run(isotropic, ms=(0, 1, 2))
    for m in (1, 2):
        bm, err = res["mspec"][m]["bm"], res["mspec"][m]["bm_err"]
        assert np.abs(bm.real / err).max() < 5.0
        assert np.abs(bm.imag / err).max() < 5.0


def test_reports_every_requested_multipole(result):
    assert sorted(result["mspec"]) == [0, 1, 2]
    nb = result["cents"].size
    assert result["mspec"][0]["cl"].shape == (nb,)
    # m=0 carries no imaginary part to speak of
    assert np.abs(result["mspec"][0]["cl"].imag).max() < 1e-6 * np.abs(result["cl_td"]).max()
    for m in (1, 2):
        assert result["mspec"][m]["bm"].shape == (nb,)
        assert np.iscomplexobj(result["mspec"][m]["bm"])


def test_quadrupole_carries_the_injected_position_angle(result):
    """The recovered b_2 phase must be exp(-2i PA), the astronomical
    convention `harmonic.beam_multipole` is written in. A mirrored local
    frame anywhere in the chain would land this at +2 PA instead."""
    bm, err = result["mspec"][2]["bm"], result["mspec"][2]["bm_err"]
    hit = np.abs(bm) > 5.0 * err
    assert hit.sum() >= 2
    # b_2 for an ellipse is negative-real at PA=0, hence the extra pi
    ang = np.angle(bm[hit] * np.exp(2j * np.deg2rad(PA)))
    assert np.allclose(np.abs(ang), np.pi, atol=0.3)


def test_dipole_is_dominated_by_the_injected_offset(result):
    """A northward offset gives a purely imaginary, negative b_1."""
    bm, err = result["mspec"][1]["bm"], result["mspec"][1]["bm_err"]
    hit = np.abs(bm) > 5.0 * err
    assert hit.sum() >= 2
    assert np.all(bm[hit].imag < 0)
    assert np.abs(bm[hit].imag).sum() > 3.0 * np.abs(bm[hit].real).sum()


def test_ms_must_contain_zero(anisotropic):
    with pytest.raises(ValueError, match="must contain 0"):
        run(anisotropic, ms=(1, 2))


def test_lmax_beyond_half_nyquist_is_refused(anisotropic):
    nyq = int(np.pi / np.abs(np.deg2rad(anisotropic["imap"].wcs.wcs.cdelt[0])))
    with pytest.raises(ValueError, match="Nyquist"):
        beams.estimate_jitter_beam(
            anisotropic["imap"],
            anisotropic["mask"],
            np.ones(nyq + 2),
            anisotropic["ras"],
            anisotropic["decs"],
            ms=(0,),
            verbose=False,
        )


def test_verbose_controls_stdout(anisotropic, capsys):
    """verbose=False must be genuinely silent, which means quietening
    pywiggle's own chatter too, not just soma's prints."""
    run(anisotropic, ms=(0,), dell=200, verbose=False)
    assert capsys.readouterr().out == ""
    run(anisotropic, ms=(0,), dell=200, verbose=True)
    assert "catalog objects inside the binary mask" in capsys.readouterr().out
