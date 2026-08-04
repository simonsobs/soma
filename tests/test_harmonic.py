"""pytest suite for soma.harmonic: azimuthal modes of a generic image."""

import numpy as np
import pytest
from pixell import enmap, utils

from soma import harmonic, maps

N, RES = 256, 0.5


def geom(n=N, res=RES, proj="tan"):
    return enmap.geometry(pos=[0, 0], res=res * utils.arcmin, shape=(n, n), proj=proj)


def rtheta(shape, wcs, center=None):
    dy, dx = maps.real_grid(shape, wcs, center=center)
    return np.hypot(dy, dx), np.arctan2(dy, dx)


def centred(arr, wcs):
    return enmap.enmap(arr, wcs)


CENTRE = (N // 2, N // 2)


def test_isotropic_image_has_only_m0():
    """An exponential profile: isotropic, and pointedly not a Gaussian beam."""
    shape, wcs = geom()
    r, _ = rtheta(shape, wcs)
    img = centred(np.exp(-r / (3.0 * utils.arcmin)), wcs)
    res = harmonic.azimuthal_modes(img, ell=np.linspace(200.0, 4000.0, 30), center=CENTRE)
    rho, _ = harmonic.mode_metrics(res["a_m"])
    for m in range(1, 7):
        assert np.nanmedian(rho[m]) < 1e-5, f"m={m}"


def test_a_hard_edge_carries_a_real_m4():
    """The input has to be isotropic, not merely radial in the formula.

    A top-hat disc is a staircase once pixelized, and the staircase has
    the grid's own four-fold symmetry, so its m=4 (1.6e-2 here) is signal
    rather than estimator error -- three orders above the same map's
    smooth-profile floor. Worth pinning: it is the one input that looks
    circularly symmetric and is not.
    """
    shape, wcs = geom()
    r, _ = rtheta(shape, wcs)
    ell = np.linspace(200.0, 4000.0, 30)
    disc = centred((r < 8.0 * utils.arcmin).astype(float), wcs)
    rho, _ = harmonic.mode_metrics(harmonic.azimuthal_modes(disc, ell=ell, center=CENTRE)["a_m"])
    assert np.nanmedian(rho[4]) > 1e-3
    assert np.nanmedian(rho[1]) < 1e-6  # odd m still vanish: it is inversion-symmetric


@pytest.mark.parametrize("m,phase", [(1, 0.0), (2, 0.4), (3, -0.9), (5, 1.3)])
def test_an_injected_angular_harmonic_comes_back_at_the_right_m_and_phase(m, phase):
    """cos(m(theta - phase)) on a ring-shaped support, recovered."""
    shape, wcs = geom()
    r, th = rtheta(shape, wcs)
    env = np.exp(-0.5 * ((r - 6.0 * utils.arcmin) / (1.5 * utils.arcmin)) ** 2)
    img = centred(env * (1.0 + 0.3 * np.cos(m * (th - phase))), wcs)
    res = harmonic.azimuthal_modes(img, ell=np.linspace(500.0, 5000.0, 40), center=CENTRE)
    rho, _ = harmonic.mode_metrics(res["a_m"])
    # the injected m dominates every other one
    others = [np.nanmedian(rho[k]) for k in range(1, 7) if k != m]
    assert np.nanmedian(rho[m]) > 20 * max(others)
    # and its axis is where it was put, modulo the pattern's own symmetry
    got = harmonic.mode_orientation(res["a_m"], m)
    period = 2 * np.pi / m
    err = np.abs((got - phase + period / 2) % period - period / 2)
    assert np.nanmedian(err) < 1e-3


def test_components_are_decomposed_independently():
    """Without qu_to_eb, an (ncomp, Ny, Nx) stack is just ncomp images."""
    shape, wcs = geom()
    r, th = rtheta(shape, wcs)
    env = np.exp(-0.5 * (r / (3.0 * utils.arcmin)) ** 2)
    a = env * (1.0 + 0.2 * np.cos(2 * th))
    b = env * (1.0 + 0.2 * np.cos(4 * th))
    ell = np.linspace(500.0, 5000.0, 20)
    stack = harmonic.azimuthal_modes(centred(np.stack([a, b, env]), wcs), ell=ell, center=CENTRE)
    assert stack["a_m"].shape == (3, 7, ell.size)
    for i, one in enumerate((a, b, env)):
        alone = harmonic.azimuthal_modes(centred(one, wcs), ell=ell, center=CENTRE)
        assert np.allclose(stack["a_m"][i], alone["a_m"], equal_nan=True)


def test_qu_to_eb_needs_three_components_and_rotates_them():
    """The spin-2 rotation is opt-in and E/B differ from raw Q/U."""
    shape, wcs = geom()
    r, th = rtheta(shape, wcs)
    p = (r / (3.0 * utils.arcmin)) ** 2 * np.exp(-0.5 * (r / (3.0 * utils.arcmin)) ** 2)
    tqu = centred(
        np.stack(
            [
                np.exp(-0.5 * (r / (3.0 * utils.arcmin)) ** 2),
                -p * np.cos(2 * th),
                -p * np.sin(2 * th),
            ]
        ),
        wcs,
    )
    ell = np.linspace(500.0, 5000.0, 20)
    raw = harmonic.azimuthal_modes(tqu, ell=ell, center=CENTRE)
    rot = harmonic.azimuthal_modes(tqu, ell=ell, center=CENTRE, qu_to_eb=True)
    assert np.allclose(raw["a_m"][0], rot["a_m"][0], equal_nan=True)  # T untouched
    # a radial pattern is pure E: the rotation moves its m=+-2 Q/U content
    # into the E monopole, which raw Q/U does not have
    assert np.nanmax(np.abs(rot["a_m"][1, 0])) > 1e3 * np.nanmax(np.abs(raw["a_m"][1, 0]))
    with pytest.raises(ValueError):
        harmonic.azimuthal_modes(centred(np.stack([tqu[0], tqu[1]]), wcs), qu_to_eb=True)


def test_rings_past_the_grid_limit_are_nan():
    shape, wcs = geom(64)
    r, _ = rtheta(shape, wcs)
    img = centred(np.exp(-0.5 * (r / (2.0 * utils.arcmin)) ** 2), wcs)
    lnyq = harmonic.l_nyquist(shape, wcs)
    dl = 2.0 * np.pi / (64 * RES * utils.arcmin)
    res = harmonic.azimuthal_modes(img, ell=np.array([lnyq, lnyq + dl]), center=CENTRE)
    assert np.isfinite(res["a_m"][0, 0]) and np.isnan(res["a_m"][0, 1])


def test_no_centering_leaves_an_off_origin_phase_ramp():
    """An off-origin feature forges modes until its phase ramp is removed.

    The offset has to be asymmetric to show up at m=1: something sitting
    exactly on (N//2, N//2) carries a parity-*even* (-1)^k checkerboard
    instead, which cannot produce odd m at all.
    """
    n = 64
    shape, wcs = geom(n)
    at = (20.0, 41.0)
    r, _ = rtheta(shape, wcs, center=at)
    img = centred(np.exp(-0.5 * (r / (0.85 * utils.arcmin)) ** 2), wcs)
    ell = np.linspace(500.0, 4000.0, 15)
    off = harmonic.azimuthal_modes(img, ell=ell, mmax=4, center=None)
    on = harmonic.azimuthal_modes(img, ell=ell, mmax=4, center=at)
    rho_off, _ = harmonic.mode_metrics(off["a_m"])
    rho_on, _ = harmonic.mode_metrics(on["a_m"])
    assert np.nanmedian(rho_off[1]) > 0.1  # forged m=1 from the ramp
    assert np.nanmedian(rho_on[1]) < 1e-12  # and gone once centred


def test_nphi_must_resolve_the_modes_asked_for():
    shape, wcs = geom(64)
    img = centred(np.ones(shape), wcs)
    with pytest.raises(ValueError, match="cannot resolve"):
        harmonic.azimuthal_modes(img, mmax=6, nphi=8)


def test_mode_floor_sits_under_a_real_signal():
    shape, wcs = geom()
    r, th = rtheta(shape, wcs)
    env = np.exp(-0.5 * (r / (3.0 * utils.arcmin)) ** 2)
    img = centred(env * (1.0 + 0.1 * np.cos(2 * th)), wcs)
    res = harmonic.azimuthal_modes(img, ell=np.linspace(1000.0, 5000.0, 20), center=CENTRE)
    rho, _ = harmonic.mode_metrics(res["a_m"])
    assert np.nanmax(harmonic.mode_floor(res["a_m"]) / rho[2]) < 1e-2


# ---------------------------------------------------------------------------
# the inverse: a_m(ell) -> image
# ---------------------------------------------------------------------------
def smooth_multipole_image(n=N, pix=0.25):
    """A Gaussian carrying m=2 and m=4, each weighted by r^m so it is
    smooth at the origin -- see `test_a_cusp_puts_power_where_rings_cannot_see`
    for what happens when it is not."""
    from soma import beams

    return beams.simulate_beam((n, n), pix, 2.0, moments={2: 0.2, 4: 0.1})


def decompose(img, mmax=8, nell=600):
    ell = np.linspace(0.0, harmonic.l_nyquist(img.shape, img.wcs), nell)
    c = (img.shape[-2] // 2, img.shape[-1] // 2)
    return harmonic.azimuthal_modes(img, ell=ell, mmax=mmax, center=c), ell, c


def test_round_trip_returns_the_image():
    img = smooth_multipole_image()
    res, ell, c = decompose(img)
    back = harmonic.azimuthal_image(res["a_m"], ell, img.shape, img.wcs, center=c)
    assert isinstance(back, enmap.ndmap) and back.dtype == np.float64
    assert back.shape == img.shape
    err = np.abs(np.asarray(back - img)).max() / np.abs(np.asarray(img)).max()
    assert err < 1e-3, err


def test_round_trip_is_the_identity_on_the_modes():
    """forward(inverse(a_m)) == a_m, for the m that carry anything."""
    img = smooth_multipole_image()
    res, ell, c = decompose(img)
    back = harmonic.azimuthal_image(res["a_m"], ell, img.shape, img.wcs, center=c)
    again = harmonic.azimuthal_modes(back, ell=ell, mmax=8, center=c)
    for m in (0, 2, 4):
        a, b = res["a_m"][m], again["a_m"][m]
        # only where the mode is really present: m=1, 3 sit at 1e-14 here,
        # so a relative comparison against them measures nothing
        sel = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 1e-3 * np.nanmax(np.abs(a)))
        assert np.median(np.abs((b[sel] - a[sel]) / a[sel])) < 1e-3, f"m={m}"


def test_multi_component_round_trip():
    from soma import beams

    img = beams.simulate_pol_beam((N, N), 0.25, 2.0, eps=0.05)
    res, ell, c = decompose(img)
    back = harmonic.azimuthal_image(res["a_m"], ell, img.shape, img.wcs, center=c)
    assert back.shape == img.shape
    assert np.abs(np.asarray(back - img)).max() / np.abs(np.asarray(img)).max() < 1e-3


@pytest.mark.parametrize("ms,keeps,drops", [([0], (), (2, 4)), ([0, 2], (2,), (4,))])
def test_ms_filters_to_the_chosen_modes(ms, keeps, drops):
    """This is what makes the pair a filter: ms=[0] isotropizes an image."""
    img = smooth_multipole_image()
    res, ell, c = decompose(img)
    filt = harmonic.azimuthal_image(res["a_m"], ell, img.shape, img.wcs, center=c, ms=ms)
    rho, _ = harmonic.mode_metrics(harmonic.azimuthal_modes(filt, ell=ell, mmax=6, center=c)["a_m"])
    sel = ell > 2000  # where the multipoles of this beam are above the floor
    for m in keeps:
        assert np.nanmedian(rho[m][sel]) > 1e-2, f"m={m} should have survived"
    for m in drops:
        assert np.nanmedian(rho[m][sel]) < 1e-3, f"m={m} should have gone"


@pytest.mark.parametrize("at", [None, (0, 0), (40, 77)])
def test_center_places_the_reconstruction(at):
    """center undoes the one given to the forward; None leaves it on (0, 0)."""
    img = smooth_multipole_image()
    res, ell, c = decompose(img)
    back = harmonic.azimuthal_image(res["a_m"], ell, img.shape, img.wcs, center=at)
    peak = np.unravel_index(np.argmax(np.asarray(back)), back.shape)
    assert peak == ((0, 0) if at is None else tuple(at))


def test_a_cusp_puts_power_where_rings_cannot_see():
    """The one limitation no ell grid can lift.

    Rings reach l_nyquist; the corners of the Cartesian Fourier grid reach
    sqrt(2) times that. A cos(2 theta) modulation *not* weighted by r^2 is
    discontinuous at the origin and puts real power out there, and exactly
    that much comes back as round-trip error -- unchanged by sampling ell
    more finely.
    """
    shape, wcs = geom()
    r, th = rtheta(shape, wcs)
    env = np.exp(-0.5 * (r / (3.0 * utils.arcmin)) ** 2)
    cusp = centred(env * (1.0 + 0.25 * np.cos(2.0 * th)), wcs)

    lnyq = harmonic.l_nyquist(shape, wcs)
    dly, dlx = enmap.lpixshape(shape, wcs, signed=True)
    ly = ((np.arange(N) - N // 2) * dly)[:, None]
    lx = ((np.arange(N) - N // 2) * dlx)[None, :]
    fmod = np.abs(np.fft.fftshift(np.asarray(enmap.fft(cusp, normalize=False))))
    outside = fmod[np.hypot(ly, lx) > lnyq].sum() / fmod.sum()

    errs = []
    for nell in (300, 2000):
        ell = np.linspace(0.0, lnyq, nell)
        res = harmonic.azimuthal_modes(cusp, ell=ell, mmax=8, center=CENTRE)
        back = harmonic.azimuthal_image(res["a_m"], ell, shape, wcs, center=CENTRE)
        errs.append(np.abs(np.asarray(back - cusp)).max() / np.abs(np.asarray(cusp)).max())
    assert outside > 1e-2  # the cusp really does reach the corners
    assert errs[0] == pytest.approx(outside, rel=0.5)  # and that is the error
    assert errs[1] == pytest.approx(errs[0], rel=0.1)  # more ell does not help


def test_azimuthal_image_rejects_mismatched_input():
    shape, wcs = geom(64)
    a_m = np.zeros((5, 20), complex)
    with pytest.raises(ValueError, match="does not end in"):
        harmonic.azimuthal_image(a_m, np.arange(19.0), shape, wcs)
    with pytest.raises(ValueError, match="ms must lie"):
        harmonic.azimuthal_image(a_m, np.arange(20.0), shape, wcs, ms=[0, 9])


def test_analytical_tf_bins_the_filter_azimuthally():
    shape, wcs = geom()
    modlmap = enmap.modlmap(shape, wcs)
    kfilter = (modlmap > 2000.0).astype(float)
    edges = np.arange(500.0, 8000.0, 500.0)
    cents, tf = harmonic.analytical_tf(modlmap, kfilter, edges)
    assert cents.size == tf.size
    # far below the cut the filter is 0, far above it is 1
    assert np.allclose(tf[cents < 1500.0], 0.0)
    assert np.allclose(tf[cents > 2500.0], 1.0)
