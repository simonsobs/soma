"""
Theory-level helpers.
"""

import numpy as np
from pixell import powspec
from scipy.interpolate import interp1d
from scipy.special import jv

__all__ = [
    "xi_from_cl",
]

_SPECS = ("TT", "EE", "BB", "TE")


def _hankel_table(cl, order, rgrid, lblock=2048):
    """Tabulate (1/2pi) sum_l l C_l J_order(l r) on ``rgrid`` (radians).

    Small-angle limit of the spin-2 Legendre sums (d^l_2,-2 and d^l_20),
    which pixell's scalar ``powspec.spec2corr`` does not provide.
    """
    ells = np.arange(cl.size, dtype=np.float64)
    out = np.zeros(rgrid.size)
    for i0 in range(0, cl.size, lblock):
        lsl = slice(i0, min(i0 + lblock, cl.size))
        out += jv(order, np.outer(rgrid, ells[lsl])) @ (ells[lsl] * cl[lsl])
    return out / (2.0 * np.pi)


def xi_from_cl(cl, rmax_rad, nr=4096):
    """Tabulate the real-space correlation functions of 1D power spectra.

    The spin-0-like functions are exact curved-sky Legendre sums computed
    with ``pixell.powspec.spec2corr``; the spin-2 combinations, for which
    pixell has no equivalent (``spec2corr`` is scalar-only, i.e. d^l_00),
    use their small-angle Hankel limits (d^l_2,-2 -> J4, d^l_20 -> J2;
    accurate to a few times 1e-5 on sub-degree scales):

    - ``tt``    = sum_l (2l+1)/(4pi) C_l^TT P_l(cos r)
    - ``plus``  = sum_l (2l+1)/(4pi) (C_l^EE + C_l^BB) P_l(cos r)
    - ``minus`` = (1/2pi) sum_l l (C_l^EE - C_l^BB) J4(l r)
    - ``cross`` = (1/2pi) sum_l l C_l^TE J2(l r)


    Parameters
    ----------
    cl : ndarray or dict
        A 1D TT spectrum (index = multipole), or a dict with key ``TT``
        and optionally ``EE``, ``BB``, ``TE`` (missing spectra are zero).
    rmax_rad : float
        Largest separation to tabulate; separations beyond it evaluate to
        0, so it must exceed the stamp diagonal.
    nr : int
        Number of radial samples.

    Returns
    -------
    dict
        Interpolator per key: ``tt`` always, and ``plus``/``minus``/
        ``cross`` when any polarization spectrum is present.
    """
    if not isinstance(cl, dict):
        cl = {"TT": cl}
    cl = {k: np.asarray(v, dtype=np.float64) for k, v in cl.items() if v is not None}
    if "TT" not in cl:
        raise ValueError("cl must contain a 'TT' spectrum")
    bad = set(cl) - set(_SPECS)
    if bad:
        raise ValueError(f"unknown spectra {sorted(bad)}; expected a subset of {_SPECS}")

    rgrid = np.linspace(0.0, rmax_rad, nr)
    lmax = max(v.size for v in cl.values())

    def pad(name):
        out = np.zeros(lmax)
        v = cl.get(name)
        if v is not None:
            out[: v.size] = v
        return out

    def legendre(spec):
        return powspec.spec2corr(spec[None, None], rgrid)[0, 0]

    tables = {"tt": legendre(pad("TT"))}
    if any(k in cl for k in ("EE", "BB", "TE")):
        ee, bb, te = pad("EE"), pad("BB"), pad("TE")
        tables["plus"] = legendre(ee + bb)
        tables["minus"] = _hankel_table(ee - bb, 4, rgrid)
        tables["cross"] = _hankel_table(te, 2, rgrid)
    return {k: interp1d(rgrid, v, bounds_error=False, fill_value=0.0) for k, v in tables.items()}
