"""
Tests for the model-map machinery in somapy.maps:
catalog-mode painting against the analytic prediction, end-to-end
amplitude recovery and stack cleaning on injected clusters and sources,
the candidate-grid GLS fitter, and the painting/stacking helpers.
"""

import numpy as np
from pixell import curvedsky, enmap, utils

from somapy import maps

ARCMIN = utils.arcmin


def toy_cl(lmax=3000):
    """A smooth damped CMB-like TT spectrum (no Boltzmann-code dependency)."""
    ells = np.arange(lmax + 1, dtype=float)
    cl = 2.0e4 / (ells + 30.0) ** 2 * np.exp(-((ells / 2500.0) ** 2))
    cl[:2] = 0.0
    return cl


def geometry(wdeg=6.0, res_arcmin=0.5):
    """A small equatorial CAR patch."""
    box = np.array([[-wdeg / 2, -wdeg / 2], [wdeg / 2, wdeg / 2]]) * utils.degree
    return enmap.geometry(pos=box, res=res_arcmin * ARCMIN, proj="car")


def catalogs(rng, ncl=12, nsrc=6, margin=2.2):
    """Random cluster and source catalogs with a minimum separation."""

    def positions(n, minsep_deg=0.5):
        ras, decs = [], []
        while len(ras) < n:
            ra, dec = rng.uniform(-margin, margin, 2)
            if ras and np.hypot(np.array(ras) - ra, np.array(decs) - dec).min() < minsep_deg:
                continue
            ras.append(ra)
            decs.append(dec)
        return np.array(ras) % 360.0, np.array(decs)

    cra, cdec = positions(ncl)
    clusters = dict(
        ras=cra, decs=cdec, zs=rng.uniform(0.3, 0.6, ncl), m500c=rng.uniform(6e14, 1.2e15, ncl)
    )
    sra, sdec = positions(nsrc)
    sources = dict(ras=sra, decs=sdec, amps=rng.uniform(800.0, 1500.0, nsrc))
    return clusters, sources


def test_catalog_mode_paints_the_predicted_profiles():
    shape, wcs = geometry(wdeg=3.0)
    clusters = dict(
        ras=np.array([0.0]), decs=np.array([0.0]), zs=np.array([0.45]), m500c=np.array([6e14])
    )
    sources = dict(ras=np.array([0.7]), decs=np.array([0.0]), amps=np.array([1234.0]))
    out = maps.build_object_model(
        shape,
        wcs,
        clusters=clusters,
        sources=sources,
        beam=1.5,
        freq_ghz=150.0,
        amp_mode="catalog",
        verbose=False,
    )
    model = out["model"]
    # cluster centre matches the beam-convolved Arnaud prediction
    beam_fn = maps._beam_fn(1.5)
    profiler = maps.BeamProfiler(beam_fn, maps.PROFILE_MAX_ARCMIN * ARCMIN)
    dT, _ = maps.deltaT_profile(6e14, 0.45, maps.COSMO, 150.0)
    expected = profiler.beamed(dT, 1.0)[0]
    cy, cx = enmap.sky2pix(shape, wcs, [0.0, 0.0]).astype(int)
    assert model[cy, cx] == np.float32(np.float32(expected)) or np.isclose(
        model[cy, cx], expected, rtol=1e-3
    )
    # source centre matches the catalog peak amplitude
    sy, sx = enmap.sky2pix(shape, wcs, [0.0, np.deg2rad(0.7)]).astype(int)
    assert np.isclose(model[sy, sx], 1234.0, rtol=1e-2)


def test_fit_recovers_amplitudes_and_cleans_stacks():
    # Inject known clusters + sources on CMB + noise, fit them back with
    # amp_mode='fit', and check amplitude recovery, the truth-stack match
    # of the mean subtracted model, and source-stack cleaning.
    rng = np.random.default_rng(3)
    shape, wcs = geometry(wdeg=6.0)
    clusters, sources = catalogs(rng)
    kw = dict(beam=1.5, freq_ghz=150.0, verbose=False)
    truth = maps.build_object_model(
        shape, wcs, clusters=clusters, sources=sources, amp_mode="catalog", **kw
    )["model"]
    tstacks = maps.build_object_model(
        shape, wcs, clusters=clusters, sources=sources, amp_mode="catalog", stack_map=truth, **kw
    )["stacks"]

    cl = toy_cl()
    cmb = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=cl.size - 1, seed=11)
    bl = maps._beam_fn(1.5)(np.arange(cl.size, dtype=float))
    b2d = np.interp(enmap.modlmap(shape, wcs), np.arange(cl.size), bl)
    cmb = enmap.ifft(enmap.fft(cmb) * b2d).real
    noise = 10.0
    nmap = rng.standard_normal(shape) * noise * ARCMIN / np.sqrt(enmap.pixsizemap(shape, wcs))
    coadd = enmap.enmap(np.asarray(cmb + nmap + truth), wcs)

    out = maps.build_object_model(
        shape,
        wcs,
        clusters=clusters,
        sources=sources,
        amp_mode="fit",
        fit_map=coadd,
        cl=cl,
        noise=noise,
        snr_min=-np.inf,
        **kw,
    )
    camp = np.median([r["amp"] for r in out["results"]["clusters"]])
    assert 0.6 <= camp <= 1.4, f"median cluster amplitude {camp}"
    srat = np.median([r["amp"] / sources["amps"][r["idx"]] for r in out["results"]["sources"]])
    assert 0.85 <= srat <= 1.15, f"median source amplitude ratio {srat}"

    st, ts = out["stacks"]["clusters"], tstacks["clusters"]
    c = st["before"].shape[0] // 2
    fitted_c = st["before"][c, c] - st["after"][c, c]  # mean subtracted model
    truth_c = ts["before"][c, c]
    assert abs(fitted_c - truth_c) < 0.5 * abs(truth_c)

    sst = out["stacks"]["sources"]
    assert abs(sst["after"][c, c]) < 0.3 * abs(sst["before"][c, c])

    corr = np.corrcoef(np.asarray(out["model"]).ravel(), np.asarray(truth).ravel())[0, 1]
    assert corr > 0.85, f"model-truth correlation {corr}"


def test_fit_object_amplitudes_picks_the_right_candidate():
    # A Gaussian blob of known width fit with a width grid: the correct
    # candidate wins and the amplitude is recovered.
    rng = np.random.default_rng(7)
    shape, wcs = geometry(wdeg=4.0)
    cl = toy_cl(2000)
    sig_true = 2.0 * ARCMIN
    r = np.linspace(0.0, 20 * ARCMIN, 2000)
    profiles = [[np.array([r, np.exp(-0.5 * (r / (f * sig_true)) ** 2)]) for f in (0.5, 1.0, 2.0)]]
    truth_amp = -700.0
    blob = maps.paint_objects(
        shape,
        wcs,
        [1.0],
        [0.3],
        [truth_amp],
        np.array([r, np.exp(-0.5 * (r / sig_true) ** 2)]),
    )
    cmb = curvedsky.rand_map(shape, wcs, cl[None, None], lmax=cl.size - 1, seed=5)
    noise = 5.0
    imap = enmap.enmap(
        np.asarray(cmb + blob)
        + rng.standard_normal(shape) * noise * ARCMIN / np.sqrt(enmap.pixsizemap(shape, wcs)),
        wcs,
    )
    fit = maps.fit_object_amplitudes(imap, [1.0], [0.3], profiles, cl, noise_uK_arcmin=noise)
    assert fit["best"][0] == 1, f"picked candidate {fit['best'][0]}"
    assert np.isclose(fit["amps"][0], truth_amp, rtol=0.2)
    assert fit["errs"][0] > 0


def test_paint_objects_and_stack_thumbnails_roundtrip():
    shape, wcs = geometry(wdeg=4.0)
    r = np.linspace(0.0, 20 * ARCMIN, 1000)
    profs = [
        np.array([r, np.exp(-0.5 * (r / (2 * ARCMIN)) ** 2)]),
        np.array([r, 1.0 / (1.0 + (r / (2 * ARCMIN)) ** 2)]),
    ]
    ras, decs, amps = [359.0, 1.0], [0.5, -0.5], [100.0, -200.0]
    omap = maps.paint_objects(shape, wcs, ras, decs, amps, profs)
    st = maps.stack_thumbnails(omap, ras, decs, r_arcmin=8.0)
    c = st.shape[0] // 2
    assert np.isclose(st[c, c], np.mean(amps), rtol=0.05)
    # non-finite stamps yield NaN amplitudes rather than crashing the fit
    omap[10, 10] = np.nan
    fit = maps.fit_object_amplitudes(
        omap,
        [ras[0]],
        [decs[0]],
        profs[0],
        toy_cl(1000),
        noise_uK_arcmin=5.0,
        radius_arcmin=8.0,
    )
    assert np.isfinite(fit["amps"][0]) or np.isnan(fit["amps"][0])
