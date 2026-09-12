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


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_complex_cl_resums_in_one_call(alm, m):
    """harm2profile(cl) must equal resumming the real and imaginary halves apart.

    The docstring of multipole_cross_spectrum promises exactly this. It used
    to be false: harm2profile cast cl to a real dtype, so a complex spectrum
    lost its sin(m phi) half to a ComplexWarning and nothing else.
    """
    ras, decs = np.array([np.rad2deg(PH0)]), np.array([90.0 - np.rad2deg(TH0)])
    cl = harmonic.multipole_cross_spectrum(alm, *harmonic.catalog_spin_alm(ras, decs, LMAX, m))
    split = harmonic.harm2profile(cl.real, BETAS, m=m) + 1j * harmonic.harm2profile(
        cl.imag, BETAS, m=m
    )
    one = harmonic.harm2profile(cl, BETAS, m=m)
    assert np.iscomplexobj(one)
    assert np.abs(one - split).max() / np.abs(split).max() < 1e-14
    # guard against the test passing because the imaginary half is negligible
    assert np.abs(split.imag).max() > 0.1 * np.abs(split.real).max()


def test_real_cl_stays_real():
    """The complex path must not have made real input return complex."""
    prof = harmonic.harm2profile(beams.gaussian_bl(np.arange(501), 5.0), BETAS, m=0)
    assert not np.iscomplexobj(prof)


@pytest.mark.parametrize("m", [-1, -2])
def test_negative_m_is_rejected(m):
    """Negative m used to return the +|m| answer -- right modulus, conjugated."""
    ras, decs = np.array([137.0]), np.array([20.0])
    with pytest.raises(ValueError, match="must be >= 0"):
        harmonic.catalog_spin_alm(ras, decs, LMAX, m)
    with pytest.raises(ValueError, match="must be >= 0"):
        harmonic.harm2profile(np.ones(LMAX + 1), BETAS, m=m)


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


@pytest.mark.parametrize("s", [0, 1, 2, 3, 4])
def test_ducc_spin_sign_convention(s):
    """The ducc convention behind catalog_spin_alm's sign correction.

    A unit E coefficient at (l, 0) synthesizes to -(-1)^s sqrt((2l+1)/4pi) d^l_{s0} for s > 0
    (and to +sqrt((2l+1)/4pi) d^l_{00} for s = 0), with d the Condon-Shortley Wigner-d of
    pywiggle that harm2profile resums with.
    """
    from pywiggle import core as pcore

    lmax, theta = 12, np.array([0.3, 0.9, 1.7, 2.5])
    loc = np.column_stack([theta, np.zeros_like(theta)])
    d = pcore.compute_wigner_d_matrix(lmax, s, 0, np.cos(theta))
    sign = 1.0 if s == 0 else -((-1.0) ** s)
    for ell in range(s, lmax + 1):
        alm = np.zeros((1 if s == 0 else 2, (lmax + 1) * (lmax + 2) // 2), complex)
        alm[0, ell] = 1.0
        q = ducc0.sht.experimental.synthesis_general(
            alm=alm, loc=loc, spin=s, lmax=lmax, epsilon=1e-12
        )[0]
        assert np.allclose(q, sign * np.sqrt((2 * ell + 1) / (4 * np.pi)) * d[:, ell], atol=1e-10)
