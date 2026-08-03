"""pytest suite for soma.stats: bandpower binning and Knox errors."""

import numpy as np
import pytest

from soma import stats


def test_bin_spectrum_is_the_mean_over_each_bin():
    cl = np.arange(100.0)
    edges = np.array([0, 10, 50, 100])
    cents, binned = stats.bin_spectrum(cl, edges)
    assert np.allclose(cents, [5.0, 30.0, 75.0])
    assert np.allclose(
        binned, [np.arange(0, 10).mean(), np.arange(10, 50).mean(), np.arange(50, 100).mean()]
    )


def test_bin_spectrum_edges_are_half_open():
    """low <= ell < high, so the boundary belongs to the upper bin."""
    cl = np.zeros(10)
    cl[5] = 1.0
    _, binned = stats.bin_spectrum(cl, np.array([0, 5, 10]))
    assert binned[0] == 0.0
    assert binned[1] == pytest.approx(1.0 / 5.0)


def test_bin_spectrum_leaves_a_constant_alone():
    cl = np.full(200, 3.5)
    _, binned = stats.bin_spectrum(cl, np.arange(0, 200, 20))
    assert np.allclose(binned, 3.5)


def test_knox_errors_scale_as_one_over_sqrt_modes():
    cents = np.array([100.0, 400.0])
    edges = np.array([50.0, 150.0, 650.0])
    xx = yy = xy = np.ones(2)
    full = stats.knox_errors(cents, edges, xx, yy, xy, 1.0)
    half = stats.knox_errors(cents, edges, xx, yy, xy, 0.25)
    # fsky enters the mode count linearly, so the error goes as 1/sqrt(fsky)
    assert np.allclose(half, 2.0 * full)


def test_knox_errors_use_both_autos_and_the_cross():
    cents, edges = np.array([100.0]), np.array([50.0, 150.0])
    a = stats.knox_errors(cents, edges, np.array([4.0]), np.array([1.0]), np.array([0.0]), 1.0)
    b = stats.knox_errors(cents, edges, np.array([4.0]), np.array([1.0]), np.array([2.0]), 1.0)
    # var = (xx*yy + xy^2)/nmodes, so a nonzero cross strictly inflates it
    assert b > a
    assert np.allclose(b**2 - a**2, 4.0 / ((2 * 100.0 + 1.0) * 100.0))


def test_bin1D_agrees_with_bin_spectrum():
    rng = np.random.default_rng(0)
    cl = rng.standard_normal(300)
    edges = np.arange(0.0, 300.0, 40.0)
    cents1, b1 = stats.bin1D(edges).bin(np.arange(cl.size, dtype=float), cl)
    cents2, b2 = stats.bin_spectrum(cl, edges)
    assert np.allclose(cents1, cents2)
    assert np.allclose(b1, b2)


def test_bin1D_ignores_nans():
    cl = np.ones(100)
    cl[3] = np.nan
    _, binned = stats.bin1D(np.array([0.0, 50.0, 100.0])).bin(np.arange(100.0), cl)
    assert np.allclose(binned, 1.0)


def test_bin2D_recovers_a_radial_profile():
    y, x = np.mgrid[-64:64, -64:64]
    modrmap = np.hypot(y, x).astype(float)
    data = 2.0 * modrmap + 1.0
    edges = np.arange(0.0, 60.0, 10.0)
    cents, prof = stats.bin2D(modrmap, edges).bin(data)
    assert prof.size == cents.size
    # bins are (lo, hi] (np.digitize right=True), so compare with masked means
    expected = [
        data[(modrmap > lo) & (modrmap <= hi)].mean()
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    assert np.allclose(prof, expected)


def test_bin2D_weights_reweight_the_mean():
    y, x = np.mgrid[-32:32, -32:32]
    modrmap = np.hypot(y, x).astype(float)
    data = np.ones_like(modrmap)
    data[y > 0] = 3.0
    w = np.ones_like(data)
    w[y > 0] = 0.0
    _, prof = stats.bin2D(modrmap, np.array([0.0, 10.0, 20.0])).bin(data, weights=w)
    assert np.allclose(prof, 1.0)
