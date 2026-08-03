"""pytest suite for soma.maps."""

import numpy as np
import pytest
from pixell import enmap, utils

from soma import maps


def geom(n=64, res=0.5):
    return enmap.geometry(pos=[0, 0], res=res * utils.arcmin, shape=(n, n), proj="tan")


def test_check_enmap_rejects_bare_arrays_and_wrong_shapes():
    shape, wcs = geom()
    with pytest.raises(TypeError):
        maps.check_enmap(np.zeros(shape))
    with pytest.raises(ValueError):
        maps.check_enmap(enmap.enmap(np.zeros((3,) + shape), wcs))
    maps.check_enmap(enmap.enmap(np.zeros(shape), wcs))
    maps.check_enmap(enmap.enmap(np.zeros((3,) + shape), wcs), ncomp=3)


def test_fourier_recenter_puts_an_integer_offset_back_at_the_origin():
    """A delta at pixel (y0, x0), recentred, transforms to a constant."""
    n = 32
    a = np.zeros((n, n))
    a[7, 21] = 1.0
    got = maps.fourier_recenter(np.fft.fft2(a), (7, 21), fourier=True)
    assert np.allclose(got, 1.0, atol=1e-12)


def test_fourier_recenter_moves_a_real_map():
    """Real map in, real map out, with the feature landing on pixel (0, 0)."""
    n = 32
    a = np.zeros((n, n))
    a[7, 21] = 1.0
    got = maps.fourier_recenter(a, (7, 21))
    assert got.dtype == a.dtype
    assert np.unravel_index(np.argmax(np.abs(got)), got.shape) == (0, 0)
    assert got[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_the_two_paths_agree():
    """fourier=True is the same operation minus the round trip."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(48, 40))
    d = (-1.7, 2.3)
    direct = maps.fourier_recenter(a, d)
    viafft = np.fft.ifft2(maps.fourier_recenter(np.fft.fft2(a), d, fourier=True)).real
    assert np.max(np.abs(direct - viafft)) < 1e-13


def test_an_integer_recentre_is_a_plain_roll():
    """Whole-pixel offsets must reproduce np.roll exactly."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(32, 32))
    got = maps.fourier_recenter(a, (3, -5))
    assert np.max(np.abs(got - np.roll(a, (-3, 5), axis=(0, 1)))) < 1e-13


def test_recentring_is_cyclic():
    """A DFT translation wraps; content past one edge returns at the other."""
    a = np.zeros((16, 16))
    a[1, 1] = 1.0
    got = maps.fourier_recenter(a, (4, 4))
    assert np.unravel_index(np.argmax(np.abs(got)), got.shape) == (13, 13)


def test_fourier_recenter_is_exact_for_fractional_offsets():
    """Round trip: recentre by d, then by -d, and get the original back."""
    rng = np.random.default_rng(0)
    bl = rng.normal(size=(48, 40)) + 1j * rng.normal(size=(48, 40))
    d = (3.7, -2.4)
    once = maps.fourier_recenter(bl, d, fourier=True)
    back = maps.fourier_recenter(once, (-d[0], -d[1]), fourier=True)
    assert np.max(np.abs(back - bl)) < 1e-13


def test_fourier_recenter_keeps_the_geometry_and_the_component_axis():
    shape, wcs = geom()
    bl = enmap.fft(enmap.enmap(np.zeros((3,) + shape), wcs), normalize=False)
    got = maps.fourier_recenter(bl, (1.5, -0.5), fourier=True)
    assert isinstance(got, enmap.ndmap)
    assert str(got.wcs) == str(bl.wcs)
    assert got.shape == bl.shape  # the leading component axis is untouched
    # the geometry stays put; only the content moves inside the pixel grid
    shape, wcs = geom()
    m = enmap.enmap(np.zeros(shape), wcs)
    assert str(maps.fourier_recenter(m, (2.5, 1.5)).wcs) == str(wcs)


def test_fourier_recenter_is_a_pure_phase():
    """It translates; it must not touch any amplitude."""
    rng = np.random.default_rng(1)
    bl = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    got = maps.fourier_recenter(bl, (2.25, 9.5), fourier=True)
    assert np.max(np.abs(np.abs(got) - np.abs(bl))) < 1e-13


def test_real_grid_is_signed_and_centred_on_the_middle_pixel():
    """Offsets run from the WCS's own handedness, about pixel N//2."""
    n = 64
    shape, wcs = geom(n)
    dy, dx = maps.real_grid(shape, wcs)
    assert dy[n // 2, n // 2] == 0.0 and dx[n // 2, n // 2] == 0.0
    dpix = enmap.pixshape(shape, wcs, signed=True)
    assert dx[0, 1] - dx[0, 0] == pytest.approx(dpix[1])  # negative on a TAN map
    assert dy[1, 0] - dy[0, 0] == pytest.approx(dpix[0])


@pytest.mark.parametrize("center", [(0, 0), (10, 3), (7.5, 12.25)])
def test_real_grid_takes_an_arbitrary_centre(center):
    """Including fractional ones -- the grid is linear, so this is exact."""
    n = 64
    shape, wcs = geom(n)
    dy, dx = maps.real_grid(shape, wcs, center=center)
    ref_y, ref_x = maps.real_grid(shape, wcs)
    dpix = enmap.pixshape(shape, wcs, signed=True)
    shift = (np.asarray(center) - n // 2) * dpix
    assert np.allclose(dy, ref_y - shift[0])
    assert np.allclose(dx, ref_x - shift[1])


def test_centroid_pixels_is_sub_pixel_and_wrap_safe():
    n = 64
    gy, gx = np.mgrid[0:n, 0:n]
    a = np.exp(-0.5 * ((gy - 20.25) ** 2 + (gx - 41.75) ** 2) / 3.0**2)
    assert maps.centroid_pixels(a) == pytest.approx((20.25, 41.75), abs=1e-6)
    # the same object rolled onto the periodic edge gives the same answer,
    # modulo the wrap
    rolled = np.roll(a, (-20, -41), axis=(0, 1))
    y0, x0 = maps.centroid_pixels(rolled)
    assert (y0 % n, x0 % n) == pytest.approx((0.25, 0.75), abs=1e-6)


def test_centroid_threshold_rejects_a_pedestal():
    """A first moment is only unbiased for a non-negative distribution."""
    n = 128
    gy, gx = np.mgrid[0:n, 0:n]
    obj = np.exp(-0.5 * ((gy - 64.0) ** 2 + (gx - 64.0) ** 2) / 2.0**2)
    # a weak, wide, off-centre pedestal: tiny in amplitude, huge in area
    bowl = 0.01 * np.exp(-0.5 * ((gy - 100.0) ** 2 + (gx - 100.0) ** 2) / 30.0**2)
    # the pedestal carries more total flux than the object, so the plain
    # first moment lands nearer to it than to the thing being measured
    biased = maps.centroid_pixels(obj + bowl)
    fixed = maps.centroid_pixels(obj + bowl, thresh=0.05)
    assert np.hypot(*(np.array(biased) - 64.0)) > 15.0  # ~16 pixels off
    assert np.hypot(*(np.array(fixed) - 64.0)) < 0.01  # 0.0025, not 0: what
    # survives the cut is the pedestal above the threshold, not nothing
