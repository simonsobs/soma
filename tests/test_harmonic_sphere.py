"""pytest suite for the curved-sky half of soma.harmonic."""

import ducc0
import numpy as np
import pytest
from pixell import curvedsky, utils

from soma import beams, harmonic

LMAX = 300
TH0, PH0 = np.deg2rad(70.0), np.deg2rad(137.0)
BETAS = np.deg2rad(np.array([2.0, 5.0, 10.0, 20.0]))


@pytest.fixture(scope="module")
def alm():
    return curvedsky.rand_alm(1.0 / (np.arange(LMAX + 1) + 20.0) ** 2, lmax=LMAX, seed=3)


def _frame(th0, ph0):
    """Unit vectors at (colat, lon): the point, local north, local east."""
    n0 = np.array([np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)])
    north = -np.array([np.cos(th0) * np.cos(ph0), np.cos(th0) * np.sin(ph0), -np.sin(th0)])
    east = np.array([-np.sin(ph0), np.cos(ph0), 0.0])
    return n0, north, east


def ring_moments(alm, th0, ph0, betas, mmax, nphi=512):
    """Azimuthal moments about a point, straight off rings of the map.

    Azimuth is measured from local north towards local east. The ring
    points are evaluated with ducc's general synthesis, so there is no
    pixelization anywhere.
    """
    n0, north, east = _frame(th0, ph0)
    phis = 2.0 * np.pi * np.arange(nphi) / nphi
    out = np.zeros((mmax + 1, betas.size), dtype=complex)
    for ib, beta in enumerate(betas):
        v = np.cos(beta) * n0[None, :] + np.sin(beta) * (
            np.cos(phis)[:, None] * north[None, :] + np.sin(phis)[:, None] * east[None, :]
        )
        loc = np.ascontiguousarray(
            np.column_stack(
                [
                    np.arccos(np.clip(v[:, 2], -1.0, 1.0)),
                    np.arctan2(v[:, 1], v[:, 0]) % (2.0 * np.pi),
                ]
            )
        )
        vals = ducc0.sht.experimental.synthesis_general(
            alm=alm[None, :], spin=0, lmax=LMAX, loc=loc, epsilon=1e-10, nthreads=0
        )[0]
        for m in range(mmax + 1):
            out[m, ib] = (vals * np.exp(-1j * m * phis)).mean()
    return out


def estimator_moments(alm, th0, ph0, betas, mmax):
    """The same moments through the pipeline: catalog transform, cross, resum."""
    ras = np.array([np.rad2deg(ph0)])
    decs = np.array([90.0 - np.rad2deg(th0)])
    out = np.zeros((mmax + 1, betas.size), dtype=complex)
    for m in range(mmax + 1):
        aE, aB = harmonic.catalog_spin_alm(ras, decs, LMAX, m)
        cl = harmonic.multipole_cross_spectrum(alm, aE, aB)
        out[m] = (
            4.0
            * np.pi
            * (
                harmonic.harm2profile(cl.real, betas, m=m)
                + 1j * harmonic.harm2profile(cl.imag, betas, m=m)
            )
        )
    return out


@pytest.mark.parametrize("m", range(9))
def test_ring_moments_match_the_estimator(alm, m):
    direct = ring_moments(alm, TH0, PH0, BETAS, 8)[m]
    mine = estimator_moments(alm, TH0, PH0, BETAS, 8)[m]
    assert np.abs(mine - direct).max() / np.abs(direct).max() < 1e-8


def test_m0_reproduces_the_legendre_profile():
    """harm2profile at m=0 is pixell's beam_transform_to_profile."""
    bl = beams.gaussian_bl(np.arange(2001), 5.0, curved=True)
    thetas = np.deg2rad(np.linspace(0.0, 20.0, 40) / 60.0)
    ref = utils.beam_transform_to_profile(bl, thetas)
    mine = harmonic.harm2profile(bl, thetas, m=0)
    assert np.abs(mine - ref).max() / np.abs(ref).max() < 1e-10


def test_spin0_has_no_b_component():
    aE, aB = harmonic.catalog_spin_alm(np.array([10.0]), np.array([-30.0]), 64, 0)
    assert np.all(aB == 0)
    assert aE.shape == aB.shape


def test_weights_are_linear(alm):
    """The catalog transform is a weighted sum of delta functions, so
    doubling a weight doubles that object's contribution."""
    ras, decs = np.array([12.0, 200.0]), np.array([-10.0, 30.0])
    a1 = harmonic.catalog_spin_alm(ras, decs, 64, 2, weights=np.array([1.0, 1.0]))[0]
    a2 = harmonic.catalog_spin_alm(ras, decs, 64, 2, weights=np.array([2.0, 2.0]))[0]
    assert np.allclose(a2, 2.0 * a1, rtol=1e-12, atol=0)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_alpha_rotates_the_catalog_phase(m):
    """A uniform frame rotation alpha multiplies the spin-m coefficients
    by exp(i m alpha)."""
    ras, decs = np.array([12.0, 200.0]), np.array([-10.0, 30.0])
    a0E, a0B = harmonic.catalog_spin_alm(ras, decs, 64, m)
    alpha = 17.0
    aE, aB = harmonic.catalog_spin_alm(ras, decs, 64, m, alphas_deg=np.full(2, alpha))
    phase = np.exp(1j * m * np.deg2rad(alpha))
    assert np.allclose(aE + 1j * aB, (a0E + 1j * a0B) * phase, rtol=1e-10, atol=1e-14)


def test_beam_multipole_carries_the_position_angle_convention():
    """b_m must follow exp(-i m PA) for structure at position angle PA."""
    cl0 = np.full(5, 2.0)
    for m, pa_deg in [(1, 40.0), (2, 30.0), (4, 20.0)]:
        # a C^(m) whose b_m should come out at exactly this phase
        target = np.exp(-1j * m * np.deg2rad(pa_deg))
        cl_m = (target / (1j**m)) * cl0
        bm = harmonic.beam_multipole(cl_m, cl0, m)
        assert np.allclose(bm, target, rtol=1e-12, atol=1e-14)


def test_beam_multipole_cancels_a_common_amplitude():
    cl0 = np.array([3.0, 2.0, 1.0])
    cl2 = np.array([0.3 + 0.1j, 0.2 + 0.05j, 0.1 + 0.02j])
    a = harmonic.beam_multipole(cl2, cl0, 2)
    b = harmonic.beam_multipole(7.5 * cl2, 7.5 * cl0, 2)
    assert np.allclose(a, b, rtol=1e-12, atol=0)
