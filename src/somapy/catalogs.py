"""
Reading object catalogs, generating synthetic ones, and converting
fluxes to map units.
"""

import numpy as np
import pandas as pd
from pixell import utils

__all__ = ["load_catalog", "random_positions", "flux_to_peak_uK"]


def load_catalog(fname, weight="flux", flux_col="S90_pred_mJy", alpha_col=None):
    """Load a quasar catalog and return positions, weights and angles.

    Parameters
    ----------
    fname : str
        Path to a CSV catalog with columns ra_deg, dec_deg (or ra, dec)
        and optionally flux and frame-angle columns.
    weight : str
        Either 'flux' (weight sources by the flux column) or 'unit'.
    flux_col : str
        Name of the flux column used for 'flux' weighting.
    alpha_col : str or None
        Name of a column holding a per-object frame rotation angle in
        degrees (scan/parallactic angle, or an alignment angle for
        oriented stacking). None for the unoriented case.

    Returns
    -------
    ras, decs : ndarray
        Source coordinates in degrees.
    weights : ndarray
        Per-source weights (normalized to mean 1).
    fluxes : ndarray
        Flux column values (mJy) if present, else ones.
    alphas : ndarray or None
        Frame angles in degrees, or None if alpha_col is None.
    """
    df = pd.read_csv(fname)
    cols = {c.lower(): c for c in df.columns}
    try:
        rcol, dcol = cols["ra_deg"], cols["dec_deg"]
    except KeyError:
        rcol, dcol = cols["ra"], cols["dec"]
    ras = df[rcol].to_numpy(float)
    decs = df[dcol].to_numpy(float)
    fluxes = df[flux_col].to_numpy(float) if flux_col in df.columns else np.ones(ras.size)
    alphas = None if alpha_col is None else df[alpha_col].to_numpy(float)
    if weight == "flux":
        w = fluxes.copy()
    elif weight == "unit":
        w = np.ones(ras.size)
    else:
        raise ValueError(f"Unknown weight scheme '{weight}'")
    good = np.isfinite(ras) & np.isfinite(decs) & np.isfinite(w) & (w > 0)
    ras, decs, w, fluxes = ras[good], decs[good], w[good], fluxes[good]
    if alphas is not None:
        alphas = alphas[good]
    return ras, decs, w / w.mean(), fluxes, alphas


def flux_to_peak_uK(flux_mJy, beam_area_sr, freq_ghz):
    """Convert source flux densities to peak map temperatures.

    Parameters
    ----------
    flux_mJy : ndarray
        Flux densities in mJy.
    beam_area_sr : float
        Beam solid angle in steradians.
    freq_ghz : float
        Observing frequency in GHz.

    Returns
    -------
    peaks : ndarray
        Peak CMB temperature amplitudes in microkelvin.
    """
    ffact = utils.flux_factor(beam_area_sr, freq_ghz * 1e9) / 1e3  # uK -> mJy
    return np.asarray(flux_mJy) / ffact


def random_positions(n, rng=None, dec_range=None):
    """Draw `n` directions distributed uniformly over the sphere.

    Uniform on the sphere, not uniform in declination: sin(dec) is what is
    drawn flat, so the poles are not oversampled.

    Parameters
    ----------
    n : int
        Number of positions.
    rng : numpy Generator, int seed, or None
        Source of randomness. An int is used as a seed, so a test can pin
        the catalog it gets.
    dec_range : (dec_lo, dec_hi) in radians, or None
        Restrict to a declination band, still uniform in area within it.

    Returns
    -------
    pos : (n, 2) ndarray
        [dec, ra] in radians, the convention `maps.thumbnails_healpix`
        and pixell's `coordinates` take.
    """
    rng = np.random.default_rng(rng)
    lo, hi = (-np.pi / 2, np.pi / 2) if dec_range is None else dec_range
    dec = np.arcsin(rng.uniform(np.sin(lo), np.sin(hi), n))
    ra = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([dec, ra], axis=1)
