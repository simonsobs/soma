"""End-to-end test of the HEALPix thumbnail chain against a closed form.

Paint a random catalog of Gaussian sources into a HEALPix map, cut
thumbnails at the source positions, stack, and ask what transfer function
came back.
"""

import healpy as hp
import numpy as np
import pytest
from pixell import utils
from scipy import special

from soma import beams, catalogs, harmonic, maps

NSIDE, FWHM, NSRC = 512, 15.0, 200
RES = 0.5 * hp.nside2resol(NSIDE)


def healpix_pixwin(nside, ell):
    """`healpy.pixwin` interpolated onto arbitrary `ell`, refusing to
    extrapolate: past the tabulated 3*nside - 1 this returns NaN rather
    than clamping, since the flat tail looks inviting and is off by a
    factor of two within a few hundred ell.

    Note the tabulation is the *power* form sqrt(<|W|^2>), averaged over
    pixels and azimuth; a stack strictly wants the amplitude form <W>.
    """
    w = hp.pixwin(nside)
    ell = np.asarray(ell, dtype=float)
    out = np.interp(ell, np.arange(w.size), w, left=np.nan, right=np.nan)
    return np.where(ell > w.size - 1, np.nan, out)


def stack_a0(hmap, pos, ell):
    """Thumbnail, stack, and return the isotropic transfer a_0(ell)."""
    thumbs = maps.thumbnails_healpix(hmap, pos, r=45 * utils.arcmin, res=RES)
    stack = thumbs.mean(0)
    centre = (stack.shape[-2] // 2, stack.shape[-1] // 2)
    modes = harmonic.azimuthal_modes(stack, ell=ell, mmax=2, center=centre)
    return np.abs(modes["a_m"][0])


@pytest.fixture(scope="module")
def painted():
    """The same catalog painted with and without the pixel window.

    `oversample=0` samples the profile at pixel centres, which imposes no
    window at all; `oversample=3` averages 64 sub-pixels per pixel, which
    imposes it. Everything downstream is identical for the two, so their
    ratio isolates the window.
    """
    pos = catalogs.random_positions(NSRC, rng=0)
    prof = maps.paint_profile(FWHM)
    averaged = maps.paint_healpix_profile(NSIDE, pos, 1.0, prof, oversample=3)
    sampled = maps.paint_healpix_profile(NSIDE, pos, 1.0, prof, oversample=0)
    return pos, averaged, sampled


# ---------------------------------------------------------------------------
# the painter
# ---------------------------------------------------------------------------
def test_pixel_averaging_conserves_flux_and_lowers_the_peak(painted):
    """Averaging over the pixel moves flux around; it does not destroy any."""
    pos, averaged, sampled = painted
    sigma = FWHM * utils.arcmin * utils.fwhm
    exact = NSRC * 2.0 * np.pi * sigma**2  # integral of a unit-peak Gaussian
    for hmap in (averaged, sampled):
        assert hmap.sum() * hp.nside2pixarea(NSIDE) == pytest.approx(exact, rel=1e-3)
    # the window shows up in real space as a lower, broader peak
    assert sampled.max() > 0.98
    assert averaged.max() < 0.95 * sampled.max()


def test_oversampling_is_converged(painted):
    """oversample=3 is not leaving anything on the table; 4 agrees with it."""
    pos, averaged, _ = painted
    finer = maps.paint_healpix_profile(NSIDE, pos, 1.0, maps.paint_profile(FWHM), oversample=4)
    peak = averaged.max()
    assert np.abs(finer - averaged).max() < 5e-3 * peak


# ---------------------------------------------------------------------------
# the headline: the window emerges, and it is healpy's
# ---------------------------------------------------------------------------
def test_stacked_thumbnails_recover_the_healpix_pixel_window(painted):
    """(averaged stack) / (sampled stack) == healpy.pixwin.

    Both stacks travel the identical thumbnail-and-stack path, so whatever
    that path does to the profile cancels in the ratio,
    leaving only what the painting differed
    in: the pixel window.
    """
    pos, averaged, sampled = painted
    ell = np.arange(50.0, 1050.0, 50.0)
    ratio = stack_a0(averaged, pos, ell) / stack_a0(sampled, pos, ell)
    want = healpix_pixwin(NSIDE, ell)
    assert np.max(np.abs(ratio / want - 1.0)) < 5e-3
    lo = ell <= 500
    assert np.max(np.abs(ratio[lo] / want[lo] - 1.0)) < 2e-3
    # and the window is a real effect over this range, not a null test
    assert want.min() < 0.85


def test_the_window_is_not_the_whole_story(painted):
    """`thumbnails_healpix` interpolates the HEALPix map bilinearly
    (`hp.get_interp_val`), and that is a third transfer function on top of
    the beam and the pixel window.
    """
    pos, averaged, _ = painted
    ell = np.arange(100.0, 1100.0, 100.0)
    got = stack_a0(averaged, pos, ell)
    got = got / got[0]
    bl = beams.gaussian_bl(ell, FWHM)
    want = bl / bl[0] * healpix_pixwin(NSIDE, ell) / healpix_pixwin(NSIDE, ell[0])
    deficit = got / want
    assert deficit[0] == pytest.approx(1.0, abs=1e-6)  # normalized there
    assert deficit[-1] < 0.75  # a third of the signal gone by ell = 1000
    assert np.all(np.diff(deficit) < 0)  # monotonic, as a smoothing kernel is


def test_the_extra_smoothing_is_set_by_the_pixel_not_the_beam():
    """It scales with the HEALPix pixel size, which is what makes it the
    interpolation and not something about the source.

    Halving the pixel doubles the ell at which a given suppression is
    reached, so the two curves lie on top of each other in ell * pixsize.
    """
    prof = maps.paint_profile(FWHM)
    curves = {}
    for nside in (256, 512):
        pos = catalogs.random_positions(NSRC, rng=0)
        # sampled, so the pixel window is absent and only the extraction
        # kernel is left
        hmap = maps.paint_healpix_profile(nside, pos, 1.0, prof, oversample=0)
        # the kernel depends on ell * pixsize, and pixsize halves from
        # nside 256 to 512, so the finer map is probed at twice the ell
        ell = np.arange(100.0, 800.0, 100.0) * (nside / 256)
        thumbs = maps.thumbnails_healpix(
            hmap, pos, r=45 * utils.arcmin, res=0.5 * hp.nside2resol(nside)
        )
        stack = thumbs.mean(0)
        centre = (stack.shape[-2] // 2, stack.shape[-1] // 2)
        a0 = np.abs(harmonic.azimuthal_modes(stack, ell=ell, mmax=2, center=centre)["a_m"][0])
        bl = beams.gaussian_bl(ell, FWHM)
        curves[nside] = (a0 / a0[0]) / (bl / bl[0])
    # nside=256 at ell and nside=512 at 2*ell see the same kernel
    assert np.max(np.abs(curves[256] / curves[512] - 1.0)) < 0.03


# ---------------------------------------------------------------------------
# the helpers
# ---------------------------------------------------------------------------
def test_healpix_pixwin_matches_healpy_and_refuses_to_extrapolate():
    w = hp.pixwin(NSIDE)
    ell = np.arange(0, w.size, 7.0)
    assert np.allclose(healpix_pixwin(NSIDE, ell), w[ell.astype(int)])
    assert np.all(np.isnan(healpix_pixwin(NSIDE, [w.size, w.size + 500.0])))


def test_random_positions_are_uniform_over_the_sphere():
    pos = catalogs.random_positions(20000, rng=0)
    assert pos.shape == (20000, 2)
    # uniform on the sphere means sin(dec) is flat, so its mean is 0 and
    # its variance 1/3 -- a dec-uniform draw would give 1/3 - (2/pi)^2
    assert np.sin(pos[:, 0]).mean() == pytest.approx(0.0, abs=0.02)
    assert np.sin(pos[:, 0]).var() == pytest.approx(1 / 3, rel=0.03)
    assert pos[:, 1].min() >= 0.0 and pos[:, 1].max() <= 2 * np.pi


def test_random_positions_respect_a_dec_band():
    band = np.deg2rad([-10.0, 10.0])
    dec = catalogs.random_positions(5000, rng=0, dec_range=band)[:, 0]
    assert dec.min() >= band[0] and dec.max() <= band[1]
    assert np.sin(dec).mean() == pytest.approx(0.0, abs=0.02)


# ---------------------------------------------------------------------------
# forward-modelling the interpolation instead of fighting it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nside", [256, 512, 1024])
def test_the_forward_model_closes(nside):
    """stack == b_ell * w_ell * T_interp, all three factors, to 1%.

    The bilinear kernel cannot be removed by extracting more finely (see
    below), so `healpix_interp_window` predicts it instead and the product
    of the three is what a stack actually measures.
    """
    ell = np.arange(100.0, min(1300.0, 1.2 * nside), 100.0)
    pos = catalogs.random_positions(NSRC, rng=0)
    hmap = maps.paint_healpix_profile(nside, pos, 1.0, maps.paint_profile(FWHM), oversample=3)
    thumbs = maps.thumbnails_healpix(
        hmap, pos, r=45 * utils.arcmin, res=0.5 * hp.nside2resol(nside)
    )
    stack = thumbs.mean(0)
    centre = (stack.shape[-2] // 2, stack.shape[-1] // 2)
    got = np.abs(harmonic.azimuthal_modes(stack, ell=ell, mmax=2, center=centre)["a_m"][0])
    got = got / got[0]

    want = (
        beams.gaussian_bl(ell, FWHM)
        * healpix_pixwin(nside, ell)
        * maps.healpix_interp_window(nside, ell)
    )
    assert np.max(np.abs(got / (want / want[0]) - 1.0)) < 0.01


def test_interp_window_is_validated_by_the_analytic_beam_alone():
    """The window is computed from healpy's interpolation weights, so it can
    be checked against a simulation whose answer is known in closed form.

    Point-sampling the profile at pixel centres imposes no pixel window, so
    a stack of those thumbnails is just b_ell * T -- no `hp.pixwin` and no
    SHT anywhere in the comparison, only the analytic Gaussian.

    nside=1024 with a 15' beam on purpose: sampling only represents the beam
    if b_ell has died by the map's band limit, and here b_ell(3*nside) is
    9e-8. At nside=512 it is 1.7e-2, and that aliasing would contaminate
    exactly the high-ell end being tested.
    """
    nside, fwhm = 1024, 15.0
    ell = np.array([200.0, 500.0, 900.0])
    pos = catalogs.random_positions(400, rng=0)
    hmap = maps.paint_healpix_profile(nside, pos, 1.0, maps.paint_profile(fwhm), oversample=0)
    thumbs = maps.thumbnails_healpix(
        hmap, pos, r=45 * utils.arcmin, res=0.5 * hp.nside2resol(nside)
    )
    stack = thumbs.mean(0)
    centre = (stack.shape[-2] // 2, stack.shape[-1] // 2)
    got = np.abs(harmonic.azimuthal_modes(stack, ell=ell, mmax=2, center=centre)["a_m"][0])

    bl = beams.gaussian_bl(ell, fwhm)
    want = bl * maps.healpix_interp_window(nside, ell)
    # normalised away from ell=0: a_0 there is the sum over a truncated stamp,
    # which is not what b_ell(0) = 1 means
    assert np.max(np.abs((got / got[0]) / (want / want[0]) - 1.0)) < 0.01


def test_interp_window_scales_with_the_pixel_size():
    """T depends on ell * pixsize alone, so halving the pixel doubles the
    ell at which a given suppression is reached.

    Not true by construction: the window is sampled from each nside's own
    pixel geometry, so this is a statement about HEALPix, checked to the
    Monte Carlo noise of the estimate.
    """
    ell = np.arange(100.0, 1200.0, 100.0)
    coarse = maps.healpix_interp_window(256, ell)
    fine = maps.healpix_interp_window(512, 2.0 * ell)
    assert np.allclose(coarse, fine, atol=3e-3)
    assert maps.healpix_interp_window(512, 0.0) == pytest.approx(1.0)
    # and it is a real suppression over the range a beam cares about
    assert maps.healpix_interp_window(512, 1000.0) < 0.7


def test_interp_window_is_converged_in_its_sampling():
    """The Monte Carlo defaults are not marginal."""
    ell = np.array([300.0, 900.0, 1400.0])
    ref = maps.healpix_interp_window(1024, ell)
    assert np.allclose(maps.healpix_interp_window(1024, ell, nsamp=500_000), ref, atol=1e-3)
    assert np.allclose(maps.healpix_interp_window(1024, ell, nbin=4096), ref, atol=1e-4)
    assert np.allclose(maps.healpix_interp_window(1024, ell, seed=7), ref, atol=1e-3)


# ---------------------------------------------------------------------------
# is the painting really imposing THE healpix pixel window?
# ---------------------------------------------------------------------------
def geometric_pixel_window(nside, ell, npix=3000, over=4, seed=0):
    """The pixel window implied by the pixel geometry alone.

    Pixel-averaging attaches to a pixel centre the mean of the field over
    that pixel, so a stack sees the field convolved with the pixel footprint
    about its own centre. Averaged over sub-pixel phase the transform is
    < mean_children J_0(ell * |offset from centre|) >, taken over pixels.
    No field, no sources, no beam -- only where HEALPix puts its pixels.
    """
    rng = np.random.default_rng(seed)
    parents = rng.integers(0, hp.nside2npix(nside), npix)
    fac = 4**over
    kids = (parents[:, None] * fac + np.arange(fac)[None, :]).ravel()
    kt, kp = hp.pix2ang(nside * 2**over, kids, nest=True)
    pt, pp = hp.pix2ang(nside, np.repeat(parents, fac), nest=True)
    sep = hp.rotator.angdist(np.array([kt, kp]), np.array([pt, pp]))
    return np.array([special.j0(L * sep).mean() for L in np.asarray(ell, float)])


@pytest.mark.parametrize("nside", [256, 512])
def test_the_pixel_window_a_stack_sees_is_healpys(nside):
    """`hp.pixwin` is conventionally a *power* window for random fields;
    a stack measures an *amplitude* response. They need not agree.

    Computed from the geometry they do, to 0.1% for ell <= 2*nside. Above
    that they part company, reaching 0.5% at 3*nside, healpy's own limit --
    the amplitude window a stack sees is the *lower* of the two, as it must
    be: the tabulation is the power form, above the amplitude form by the
    kernel variance (pinned in the two-moments test below).
    """
    ell = np.arange(0.0, 2.0 * nside, nside / 8)
    geo = geometric_pixel_window(nside, ell)
    assert np.max(np.abs(geo / hp.pixwin(nside)[ell.astype(int)] - 1.0)) < 2e-3

    high = np.arange(2.0 * nside, 3.0 * nside, nside / 8)
    dev = geometric_pixel_window(nside, high) / hp.pixwin(nside)[high.astype(int)]
    assert dev.min() < 0.999  # they really do diverge up there
    assert dev.min() > 0.99  # but only by half a per cent


def test_the_pixel_windows_two_moments_are_close_but_not_identical():
    """`hp.pixwin` is the power form sqrt(<|W|^2>); a stack sees <W>.

    The gap between the two is the kernel's variance, and for pixels it is
    dominated by the *anisotropy* of each pixel's window (power averages
    |W|^2 over azimuth where a stack averages W) rather than pixel-to-pixel
    shape scatter. It is small -- which is why one tabulation serves both
    uses -- but not zero, and it grows toward the band limit: under 0.01%
    at ell = nside, ~0.15% at 2*nside, ~0.9% at 3*nside. Both moments are
    computed here from the same sub-pixel children, so the quadrature
    error cancels in the ratio.
    """
    nside, npix, over = 512, 200, 3
    rng = np.random.default_rng(0)
    parents = rng.integers(0, hp.nside2npix(nside), npix)
    fac = 4**over
    kids = parents[:, None] * fac + np.arange(fac)[None, :]
    kv = np.asarray(hp.pix2vec(nside * 2**over, kids.ravel(), nest=True)).reshape(3, npix, fac)
    pv = np.asarray(hp.pix2vec(nside, parents, nest=True))
    dctr = np.arccos(np.clip(np.einsum("ip,ipc->pc", pv, kv), -1, 1))
    spair = np.arccos(np.clip(np.einsum("ipc,ipd->pcd", kv, kv), -1, 1))
    gaps = []
    for x in (1.0, 2.0, 3.0):
        amp = special.j0(x * nside * dctr).mean()
        power = np.sqrt(special.j0(x * nside * spair).mean())
        gaps.append(power / amp - 1.0)
    assert gaps[0] < 3e-4  # ell = nside: negligible
    assert 1e-3 < gaps[1] < 2e-3  # 2*nside: real but an order below interp's
    assert gaps[2] > 5e-3  # 3*nside: approaching a per cent
    assert np.all(np.diff(gaps) > 0)


def test_the_painted_map_carries_that_window_and_not_something_else(painted):
    """Close the loop: what the simulation recovers is what the geometry says.

    This is the check that the catalog -> map step means what it claims. The
    window is a property of the pixelisation, so the recovered one must not
    depend on the beam that was painted -- and it does not, across a factor
    of three in FWHM.
    """
    pos, averaged, sampled = painted
    ell = np.arange(100.0, 1100.0, 100.0)
    recovered = stack_a0(averaged, pos, ell) / stack_a0(sampled, pos, ell)
    assert np.max(np.abs(recovered / geometric_pixel_window(NSIDE, ell) - 1.0)) < 6e-3

    # a different beam through the same pixels must give the same window
    other = maps.paint_profile(25.0)
    avg2 = maps.paint_healpix_profile(NSIDE, pos, 1.0, other, oversample=3)
    smp2 = maps.paint_healpix_profile(NSIDE, pos, 1.0, other, oversample=0)
    recovered2 = stack_a0(avg2, pos, ell) / stack_a0(smp2, pos, ell)
    assert np.max(np.abs(recovered2 / recovered - 1.0)) < 6e-3


def test_undersampled_oversampling_biases_the_window_high(painted):
    """oversample is not free: too few children under-resolve the pixel and
    leave the map smoother than a real pixel average, which shows up as a
    window biased *towards one*."""
    pos, _, sampled = painted
    ell = np.array([500.0, 1100.0])
    want = geometric_pixel_window(NSIDE, ell)
    prof = maps.paint_profile(FWHM)
    coarse = maps.paint_healpix_profile(NSIDE, pos, 1.0, prof, oversample=1)
    got = stack_a0(coarse, pos, ell) / stack_a0(sampled, pos, ell)
    assert np.all(got / want > 1.01)  # 4 children is visibly not enough


def test_the_interp_window_is_the_amplitude_form_not_the_power_one():
    """`healpix_interp_window` is < K >, correct for a stack, wrong for a C_ell.

    The power form is sqrt(<|K|^2>), and for interpolation the two are far
    apart because the kernel swings with sub-pixel phase: on a pixel centre
    one weight is 1 and nothing is smoothed, halfway between them the
    weights spread and smoothing is maximal. Pixel *shapes* vary hardly at
    all, which is why one `hp.pixwin` serves both purposes and this does
    not.

    <exp(i l.d)> over the direction of l is J_0(l|d|), so the power form is
    < sum_ij w_i w_j J_0(l s_ij) > with s_ij the separation between
    neighbours i and j.
    """
    nside, nsamp = 1024, 60000
    rng = np.random.default_rng(0)
    theta = np.arccos(rng.uniform(-1.0, 1.0, nsamp))
    phi = rng.uniform(0.0, 2.0 * np.pi, nsamp)
    pix, w = hp.get_interp_weights(nside, theta, phi)
    pt, pp = hp.pix2ang(nside, pix.ravel())
    pt, pp = pt.reshape(4, nsamp), pp.reshape(4, nsamp)

    ell = np.array([0.9, 1.4, 2.0]) * nside
    sep = np.empty((4, 4, nsamp))
    for i in range(4):
        for j in range(4):
            sep[i, j] = hp.rotator.angdist(np.array([pt[i], pp[i]]), np.array([pt[j], pp[j]]))
    power = np.array(
        [
            np.sqrt((w[:, None, :] * w[None, :, :] * special.j0(L * sep)).sum((0, 1)).mean())
            for L in ell
        ]
    )
    amp = maps.healpix_interp_window(nside, ell)
    excess = power / amp - 1.0
    assert excess[0] < 3e-3  # ell = 0.9 nside: still negligible
    assert excess[1] > 2e-3  # 1.4 nside: showing
    assert excess[2] > 0.02  # 2.0 nside: 3.5%, not ignorable
    assert np.all(np.diff(excess) > 0)  # and it only grows
