"""pytest suite for the object-painting and mask helpers in soma.maps.
"""

import numpy as np
import pytest
from pixell import coordinates, enmap, utils

from soma import beams, harmonic, maps

RES = 0.5


def fullsky(res=8.0):
    return enmap.fullsky_geometry(res=res * utils.arcmin)


def stamp(n=256, res=RES):
    return enmap.geometry(pos=[0, 0], res=res * utils.arcmin, shape=(n, n), proj="car")


def moment_pa(bmap, dec_deg, ra_deg, thresh=1e-6):
    """PA (deg east of north, mod 180) from second moments in the rotated local frame.

    Rotates every pixel position into the frame centred on (ra_deg,
    dec_deg) with `coordinates.recenter`, so it is exact at any
    declination and in any projection -- unlike the flat
    (dec - dec0, (ra - ra0) cos dec) frame that is good enough near the
    equator.
    """
    decs, ras = enmap.posmap(bmap.shape[-2:], bmap.wcs)
    v, u = coordinates.recenter(
        np.array([ras, decs]), [np.deg2rad(ra_deg), np.deg2rad(dec_deg), 0.0, 0.0]
    )
    v = v * np.cos(u)
    w = np.where(np.asarray(bmap) > thresh, np.asarray(bmap), 0.0)
    tot = w.sum()
    suu = (w * u * u).sum() / tot
    svv = (w * v * v).sum() / tot
    suv = (w * u * v).sum() / tot
    return np.rad2deg(0.5 * np.arctan2(2.0 * suv, suu - svv)) % 180.0


def test_paint_profile_is_a_unit_peak_gaussian():
    r, prof = maps.paint_profile(2.0)
    assert prof[0] == pytest.approx(1.0)
    sigma = 2.0 * utils.arcmin * utils.fwhm
    assert np.allclose(prof, np.exp(-0.5 * (r / sigma) ** 2))
    # half maximum sits at the half width, by construction
    assert np.interp(0.5, prof[::-1], r[::-1]) == pytest.approx(utils.arcmin, rel=1e-3)


def test_paint_sources_puts_the_peak_at_the_source():
    shape, wcs = fullsky()
    # exactly on a pixel centre, so the peak pixel samples the profile at r=0
    # rather than somewhere down its flank
    py0, px0 = 300, 900
    decs, ras = np.rad2deg(enmap.pix2sky(shape, wcs, [[py0], [px0]]))
    omap = maps.paint_sources(shape, wcs, ras, decs, np.array([100.0]), maps.paint_profile(20.0))
    assert omap[py0, px0] == pytest.approx(100.0, rel=1e-4)
    assert np.unravel_index(np.argmax(omap), omap.shape) == (py0, px0)


def test_paint_sources_is_subpixel():
    """Shifting a source by a fraction of a pixel moves the centroid by that
    fraction, rather than snapping to the nearest pixel."""
    shape, wcs = fullsky()
    decs, ras = np.rad2deg(enmap.pix2sky(shape, wcs, [[300.0], [900.0]]))
    prof = maps.paint_profile(20.0)
    pix = np.abs(wcs.wcs.cdelt[0])
    a = maps.paint_sources(shape, wcs, ras, decs, np.array([1.0]), prof)
    b = maps.paint_sources(shape, wcs, ras + 0.4 * pix, decs, np.array([1.0]), prof)
    assert not np.allclose(a, b)
    cy_a, cx_a = maps.centroid_pixels(np.asarray(a))
    cy_b, cx_b = maps.centroid_pixels(np.asarray(b))
    assert abs(cy_b - cy_a) < 1e-3
    # +RA is -x on this cdelt1 < 0 geometry
    assert cx_b - cx_a == pytest.approx(-0.4, abs=0.05)


def test_paint_elliptical_reduces_to_a_circle():
    """Equal axes must reproduce the radial painter."""
    shape, wcs = fullsky()
    ras, decs, amps = np.array([50.0]), np.array([10.0]), np.array([1.0])
    ell = maps.paint_elliptical(shape, wcs, ras, decs, amps, 20.0, 20.0, 0.0)
    rad = maps.paint_sources(shape, wcs, ras, decs, amps, maps.paint_profile(20.0))
    assert np.abs(ell - rad).max() < 2e-3


@pytest.mark.parametrize("pa_deg", [0.0, 30.0, 75.0])
def test_paint_elliptical_matches_the_closed_form_multipoles(pa_deg):
    """An elliptical Gaussian must decompose to I_1(a)/I_0(a) at m=2, with
    its axis at the position angle it was painted with."""
    shape, wcs = stamp()
    fmaj, fmin = 6.0, 4.0
    ny, nx = shape
    dec, ra = np.rad2deg(enmap.pix2sky(shape, wcs, [[ny // 2], [nx // 2]]))[:, 0]
    bmap = maps.paint_elliptical(
        shape, wcs, np.array([ra]), np.array([dec]), np.array([1.0]), fmaj, fmin, pa_deg
    )
    ell = np.linspace(200.0, 3000.0, 40)
    res = harmonic.azimuthal_modes(bmap, ell=ell, mmax=6, center=(ny // 2, nx // 2))
    rho, _ = harmonic.mode_metrics(res["a_m"])
    assert np.allclose(rho[2], beams.elliptical_rho(ell, fmaj, fmin), rtol=3e-2, atol=1e-3)
    # odd modes vanish for an ellipse
    assert np.nanmax(rho[1]) < 5e-3
    # `mode_orientation` reports the axis in the *map* frame, +x toward +y.
    # `paint_elliptical` takes a sky position angle, north toward east, and
    # east is -x on this cdelt1 < 0 geometry -- hence the 90 - PA. Getting
    # 90 + PA here would mean the painter had mirrored the local frame.
    ang = np.median(harmonic.mode_orientation(res["a_m"], 2, deg=True)) % 180.0
    assert ang == pytest.approx((90.0 - pa_deg) % 180.0, abs=2.0)


@pytest.mark.parametrize("flip", [False, True])
def test_paint_elliptical_position_angle_is_wcs_handedness_independent(flip):
    """The painted position angle must be east of north on the sky whatever
    the sign of cdelt1, which is what forces the signed pixel spacings."""
    shape, wcs = stamp()
    if flip:
        wcs = wcs.deepcopy()
        wcs.wcs.cdelt = [-wcs.wcs.cdelt[0], wcs.wcs.cdelt[1]]
    ny, nx = shape
    dec, ra = np.rad2deg(enmap.pix2sky(shape, wcs, [[ny // 2], [nx // 2]]))[:, 0]
    bmap = maps.paint_elliptical(
        shape, wcs, np.array([ra]), np.array([dec]), np.array([1.0]), 6.0, 4.0, 30.0
    )
    # second moments in a true (north, east) frame, independent of the wcs
    decs, ras = enmap.posmap(shape, wcs)
    u = decs - np.deg2rad(dec)
    v = (ras - np.deg2rad(ra)) * np.cos(decs)
    w = np.where(np.asarray(bmap) > 1e-6, np.asarray(bmap), 0.0)
    suu = (w * u * u).sum() / w.sum()
    svv = (w * v * v).sum() / w.sum()
    suv = (w * u * v).sum() / w.sum()
    ang = np.rad2deg(0.5 * np.arctan2(2.0 * suv, suu - svv)) % 180.0
    assert ang == pytest.approx(30.0, abs=0.5)


def test_paint_elliptical_conserves_the_solid_angle():
    """Splitting the FWHM at fixed geometric mean leaves the integral alone."""
    shape, wcs = fullsky(res=4.0)
    ras, decs, amps = np.array([80.0]), np.array([5.0]), np.array([1.0])
    pix = enmap.pixsizemap(shape, wcs)
    tot = []
    for q in [1.0, np.sqrt(1.5), np.sqrt(2.5)]:
        m = maps.paint_elliptical(shape, wcs, ras, decs, amps, 20.0 * q, 20.0 / q, 20.0)
        tot.append(float((m * pix).sum()))
    assert np.allclose(tot, tot[0], rtol=2e-3)


def test_paint_elliptical_wraps_in_ra():
    """A source at RA=0 must not lose the half of its stamp that runs off
    the left edge of the map."""
    shape, wcs = fullsky()
    amps = np.array([1.0])
    at0 = maps.paint_elliptical(shape, wcs, np.array([0.0]), np.array([0.0]), amps, 30.0, 20.0, 0.0)
    mid = maps.paint_elliptical(
        shape, wcs, np.array([180.0]), np.array([0.0]), amps, 30.0, 20.0, 0.0
    )
    pix = enmap.pixsizemap(shape, wcs)
    assert float((at0 * pix).sum()) == pytest.approx(float((mid * pix).sum()), rel=1e-6)


@pytest.mark.parametrize("dec,atol", [(0.0, 1e-5), (40.0, 2e-3), (-62.0, 3e-3)])
def test_paint_elliptical_paths_agree(dec, atol):
    """The flat CAR frame and the general pix2sky one must paint the same
    thing, to the accuracy of the flat frame."""
    shape, wcs = fullsky(res=2.0)
    args = (shape, wcs, np.array([137.0]), np.array([dec]), np.array([1.0]), 24.0, 16.0, 20.0)
    a = maps.paint_elliptical(*args, method="car")
    b = maps.paint_elliptical(*args, method="general")
    assert np.abs(np.asarray(a) - np.asarray(b)).max() < atol
    pix = enmap.pixsizemap(shape, wcs)
    assert float((a * pix).sum()) == pytest.approx(float((b * pix).sum()), rel=1e-4)


@pytest.mark.parametrize("proj", ["tan", "zea", "car"])
@pytest.mark.parametrize("dec0,pa", [(0.0, 30.0), (60.0, 30.0), (60.0, 75.0)])
def test_paint_elliptical_position_angle_on_any_projection(proj, dec0, pa):
    """The position angle is east of north on the sky, whatever the
    projection -- which only the general path can guarantee."""
    shape, wcs = enmap.geometry(
        pos=np.deg2rad([dec0, 0.0]), res=0.5 * utils.arcmin, shape=(256, 256), proj=proj
    )
    bmap = maps.paint_elliptical(
        shape, wcs, np.array([0.0]), np.array([dec0]), np.array([1.0]), 6.0, 4.0, pa
    )
    assert moment_pa(bmap, dec0, 0.0) == pytest.approx(pa % 180.0, abs=0.5)


def test_paint_elliptical_is_not_truncated_at_high_declination():
    """An x pixel subtends |cdelt1| cos(dec), so sizing the stamp from
    cdelt alone cuts the wings in east and not in north at high dec -- a
    quadrupolar truncation, which is exactly the signal being measured."""
    shape, wcs = fullsky(res=2.0)
    pix = enmap.pixsizemap(shape, wcs)
    smaj, smin = 24.0 * utils.arcmin * utils.fwhm, 16.0 * utils.arcmin * utils.fwhm

    def flux(dec):
        m = maps.paint_elliptical(
            shape, wcs, np.array([70.0]), np.array([dec]), np.array([1.0]), 24.0, 16.0, 20.0
        )
        return float((m * pix).sum())

    assert flux(70.0) == pytest.approx(flux(0.0), rel=1e-4)
    assert flux(0.0) == pytest.approx(2.0 * np.pi * smaj * smin, rel=1e-3)


def test_paint_elliptical_does_not_wrap_on_a_cutout():
    """A cutout of a full-sky map is not periodic in its own pixel
    indices: a source near its x edge must be clipped, not wrapped."""
    fshape, fwcs = fullsky(res=4.0)
    shape, wcs = enmap.slice_geometry(fshape, fwcs, (slice(500, 800), slice(200, 600)))
    dec, ra = np.rad2deg(enmap.pix2sky(shape, wcs, [[150], [3]]))[:, 0]
    m = maps.paint_elliptical(
        shape, wcs, np.array([ra]), np.array([dec]), np.array([1.0]), 30.0, 20.0, 0.0
    )
    assert float(np.asarray(m)[:, -20:].sum()) < 1e-8 * float(m.sum())
    assert float(np.asarray(m)[:, :20].sum()) > 0.5 * float(m.sum())


def test_paint_elliptical_rejects_bad_arguments():
    shape, wcs = fullsky()
    args = (np.array([0.0]), np.array([0.0]), np.array([1.0]), 20.0, 20.0)
    with pytest.raises(ValueError):
        maps.paint_elliptical((shape[0],), wcs, *args)
    with pytest.raises(ValueError):
        maps.paint_elliptical(shape, wcs, *args, method="nope")


def test_paint_elliptical_on_a_plain_geometry_has_no_cos_dec():
    """A plain geometry's cdelt is a literal, coordinate-independent
    spacing, so the same object must paint the same flux anywhere."""
    shape, wcs = enmap.geometry(
        pos=np.deg2rad([[-5, -5], [5, 5]]), res=0.5 * utils.arcmin, proj="plain"
    )
    tot = [
        float(
            maps.paint_elliptical(
                shape, wcs, np.array([0.0]), np.array([y]), np.array([1.0]), 6.0, 4.0, 20.0
            ).sum()
        )
        for y in (0.0, 4.0)
    ]
    assert tot[1] == pytest.approx(tot[0], rel=1e-4)


def strip_mask(res=8.0, dec_lim=30.0):
    shape, wcs = fullsky(res=res)
    decs = np.rad2deg(enmap.pix2sky(shape, wcs, [np.arange(shape[0]), np.zeros(shape[0])])[0])
    strip = (np.abs(decs) < dec_lim).astype(np.float64)
    return enmap.enmap(np.repeat(strip[:, None], shape[1], axis=1), wcs)


def test_wfactor_of_a_binary_mask_is_its_sky_fraction():
    mask = strip_mask()
    # analytic: the fraction of the sphere within +-30 deg of the equator is
    # sin(30) = 0.5, and a binary mask is unchanged by any power
    for n in [1, 2, 4]:
        assert maps.wfactor(n, mask) == pytest.approx(0.5, rel=1e-3)


def test_wfactor_falls_with_the_power_for_a_graded_mask():
    shape, wcs = fullsky()
    mask = enmap.enmap(np.full(shape, 0.5), wcs)
    assert maps.wfactor(1, mask) == pytest.approx(0.5, rel=1e-6)
    assert maps.wfactor(2, mask) == pytest.approx(0.25, rel=1e-6)


def test_wfactor_fft_normalization_is_relative_to_the_map():
    """sht=True normalizes to the full sky, sht=False to the map's own area,
    so a full-sky mask makes them agree and a partial one does not."""
    mask = strip_mask()
    assert maps.wfactor(2, mask, sht=False) == pytest.approx(0.5, rel=1e-3)
    grey = enmap.enmap(np.full(mask.shape, 0.5), mask.wcs)
    assert maps.wfactor(2, grey, sht=True) == pytest.approx(0.25, rel=1e-6)
    assert maps.wfactor(2, grey, sht=False) == pytest.approx(0.25, rel=1e-6)


def test_wfactor_accepts_a_precomputed_pixel_area_map():
    mask = strip_mask()
    pmap = enmap.pixsizemap(mask.shape, mask.wcs)
    assert maps.wfactor(2, mask, pmap=pmap) == pytest.approx(maps.wfactor(2, mask), rel=1e-12)


def test_wfactor_equal_area_shortcut_matches_on_a_healpix_style_mask():
    """With equal_area the pixel area is a scalar, which is exact for
    healpix and is why 1d masks are allowed at all."""
    npix = 12 * 64**2
    mask = np.zeros(npix)
    mask[: npix // 4] = 1.0
    assert maps.wfactor(2, mask, equal_area=True) == pytest.approx(0.25, rel=1e-12)


def test_wfactor_rejects_the_wrong_dimensionality():
    with pytest.raises(AssertionError):
        maps.wfactor(2, np.ones((2, 4, 4)))


def test_cosine_apodize_none_is_a_passthrough():
    mask = strip_mask()
    assert maps.cosine_apodize(mask, None) is mask


def test_cosine_apodize_tapers_the_edge_and_leaves_the_interior():
    mask = strip_mask(res=8.0, dec_lim=30.0)
    apod = maps.cosine_apodize(mask, 5.0)
    assert apod.min() >= 0.0 and apod.max() <= 1.0 + 1e-12
    decs = np.rad2deg(enmap.posmap(mask.shape, mask.wcs)[0])
    deep = np.abs(decs) < 20.0  # well inside the taper width
    assert np.allclose(np.asarray(apod)[deep], 1.0)
    assert np.all(np.asarray(apod)[np.abs(decs) > 30.0] == 0.0)
    # the taper zone is genuinely graded, not a step
    edge = (np.abs(decs) > 26.0) & (np.abs(decs) < 29.0)
    assert 0.0 < np.asarray(apod)[edge].min() < np.asarray(apod)[edge].max() < 1.0


def test_cosine_apodize_matches_orphics():
    """Copied verbatim, so it must agree bit for bit."""
    orphics_maps = pytest.importorskip("orphics.maps")
    mask = strip_mask()
    assert np.array_equal(
        np.asarray(maps.cosine_apodize(mask, 5.0)),
        np.asarray(orphics_maps.cosine_apodize(mask, 5.0)),
    )
