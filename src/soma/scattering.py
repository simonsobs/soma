"""Directional scattering covariances on the sphere.

``ScatterTransform`` computes the statistics of ``s2scat.scatter`` (mean, variance, S1, P00,
C01 and C11, in the same order and normalisation) for a real field given its harmonic
coefficients. ``CARScatterTransform`` and ``HealpixScatterTransform`` take a CAR or a HEALPix
map instead. Each of them also takes a second field, and then returns cross statistics.

Constructing an object is the expensive step: the filters, geometries and weights of every
wavelet scale are prepared once, so reuse one object for many fields at the same band limit.

There are two backends. The numpy backend runs on the CPU. The JAX backend
(``backend="jax"``) is a jitted, differentiable function that runs wherever its input array
lives; it is optional and needs pixell and ducc built with JAX support, as described in
``examples/scattering_des.ipynb``. The environment variable ``SOMA_SCATTERING_BACKEND`` sets the
default: ``numpy``, ``jax``, or ``auto`` (the default), which chooses JAX only where it runs on
a GPU, since on a CPU both backends call the same transforms.

Outline of the transform:

* The directional wavelets of s2wav, psi^j_{ln} = kappa_j(l) s_{ln}, are built
  here in numpy. For N directions only the orders
  n < N with n + N odd are non-zero; for N = 3 these are n = 0 and n = 2.
* The wavelet coefficients W^j(gamma) for the 2N-1 orientations gamma are a
  Wigner transform, which for a real field reduces to one spin-0 and one
  spin-n synthesis per active order n, combined by a fixed steering matrix
  (``ScatterTransform.wavelet``).
* The scattering covariances are computed from |W| with a second wavelet layer
  and quadrature weighted pixel sums (``ScatterTransform.from_alm``).

Typical use::

    from soma.scattering import ScatterTransform, CARScatterTransform, HealpixScatterTransform

    st = ScatterTransform(lmax, N=3, J_min=2, mask=mask)      # input: healpy alm, any lmax
    st = CARScatterTransform(shape, wcs, mask=mask)           # input: full-sky Fejer-1 enmap
    st = HealpixScatterTransform(lmax, niter=3, mask=hpmask)  # input: HEALPix map, ring ordering
    mean, var, S1, P00, C01, C11 = st(x)
    mean, var, S1, P00, C01, C11 = st.from_alm(alm)           # alm directly, on any of the three
    mean, var, S1, P00, C01, C11 = st(x, y)                   # cross statistics of two fields
    st = ScatterTransform(lmax, backend="jax")                # jitted JAX function of the alm
    grad = jax.grad(lambda a: st(a)[5].sum())(alm)            # differentiable
    batched = jax.vmap(st.from_alm)                           # batches of fields
"""

import os
from math import comb

import numpy as np
from pixell import curvedsky, enmap

__all__ = [
    "ScatterTransform",
    "CARScatterTransform",
    "HealpixScatterTransform",
]

# ----------------------------------------------------------------------------
# harmonic coefficient helpers (healpy layout: m major, ell minor, lmax inclusive)


def _alm_ell(lmax):
    """ell of every coefficient in the healpy ordering."""
    return np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])


# ----------------------------------------------------------------------------
# s2wav / S2LET directional wavelet filters, in numpy


def _j_max(L, lam=2.0):
    return int(np.ceil(np.log(L) / np.log(lam)))


def _k_lam(L, lam=2.0, quad_iters=300):
    """Scale-discretised wavelet generating function k_lambda of Leistedt et al. (2013).

    Trapezium integration of exp(-2/(1-s^2))/t with the same rule as s2wav (intervals
    touching the end points 1/lambda and 1 are skipped), so the filters match s2wav.
    """

    def part(a, b):
        if a == b:
            return 0.0
        h = (b - a) / quad_iters
        t = a + h * np.arange(quad_iters + 1)
        keep = ~np.isin(t[:-1], [1 / lam, 1.0]) & ~np.isin(t[1:], [1 / lam, 1.0])
        x = (t - 1.0 / lam) * (2.0 * lam / (lam - 1.0)) - 1.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            f = np.exp(-2.0 / (1.0 - x**2)) / t
        return np.sum(((f[:-1] + f[1:]) * h / 2)[keep])

    J = _j_max(L, lam)
    norm = part(1.0 / lam, 1.0)
    k = np.zeros((J + 2, L))
    for j in range(J + 2):
        for ell in range(L):
            if ell < lam ** (j - 1):
                k[j, ell] = 1
            elif ell <= lam**j:
                k[j, ell] = part(ell / lam**j, 1.0) / norm
    return k


def _directional_filters(L, N, lam=2.0):
    """Directional wavelet filters psi[j, ell, n] for n = 0..N-1, identical to s2wav.

    Returns a real array of shape (J_max+1, L, N). Only n with n + N odd are non-zero
    and psi[l, -n] = psi[l, n]. The scaling function is not needed by the transform.
    """
    k = _k_lam(L, lam)
    kappa = np.sqrt(np.clip(k[1:] - k[:-1], 0, None))  # (J+1, L)
    kappa *= np.sqrt((2 * np.arange(L) + 1) / 8.0) / np.pi
    s = np.zeros((L, N))
    for ell in range(1, L):
        gamma = min(N - 1, ell) if (N + ell) % 2 else min(N - 1, ell - 1)
        for n in range(N):
            if (N + n) % 2 and n <= gamma:
                s[ell, n] = np.sqrt(comb(gamma, (gamma - n) // 2) / 2**gamma)
    if N % 2 == 0:
        raise ValueError("even N gives complex filters; this module supports odd N")
    return np.ascontiguousarray(kappa[:, :, None] * s[None])


def _active_orders(N):
    return [n for n in range(N) if (n + N) % 2 == 1]


def _steering_matrix(N):
    """(2N-1, ncomp) matrix giving W(gamma_g) from (f_0, Re f_n, Im f_n, ...)."""
    gam = 2 * np.pi * np.arange(2 * N - 1) / (2 * N - 1)
    cols = []
    for n in _active_orders(N):
        cols += [np.ones_like(gam)] if n == 0 else [2 * np.cos(n * gam), -2 * np.sin(n * gam)]
    return np.array(cols).T


# ----------------------------------------------------------------------------
# the transform


def _healpix_to_mask(hmap, shape, wcs):
    """Boolean mask on a CAR geometry from a HEALPix map (ring ordering) by nearest pixel."""
    import healpy as hp

    dec, ra = enmap.posmap(shape, wcs)
    pix = hp.ang2pix(hp.npix2nside(len(hmap)), np.pi / 2 - dec, ra % (2 * np.pi))
    return np.asarray(hmap)[pix] > 0.5


def _jax_available(device_only=False):
    """Whether JAX and a ducc build with the XLA handlers of its SHT backend are present.

    With ``device_only`` this also requires JAX to run on an accelerator rather than on a
    CPU, which is what the automatic choice of backend asks for.
    """
    try:
        import ducc0
        import jax
    except ImportError:
        return False
    if not (hasattr(ducc0.sht, "sht_ffi_targets") or hasattr(ducc0.sht, "sht_ffi_cpu_targets")):
        return False
    return jax.default_backend() != "cpu" if device_only else True


def _default_backend():
    """The backend named by ``SOMA_SCATTERING_BACKEND`` (numpy, jax or auto; default auto).

    ``auto`` picks JAX only where it runs on a GPU. On a CPU the two backends call the same
    ducc transforms, and the numpy one gets there without tracing or compiling.
    """
    b = os.environ.get("SOMA_SCATTERING_BACKEND", "auto").lower()
    if b == "auto":
        return "jax" if _jax_available(device_only=True) else "numpy"
    if b not in ("numpy", "jax"):
        raise ValueError("SOMA_SCATTERING_BACKEND must be numpy, jax or auto")
    return b


def _tick(timer, msg):
    if timer is not None:
        timer.tick(msg)


def _transfer_index(lmax_in, lmax_out):
    """Gather index and validity mask taking healpy ordered alm from lmax_in to lmax_out."""
    ell = _alm_ell(lmax_out)
    m = np.concatenate([np.full(lmax_out + 1 - mm, mm) for mm in range(lmax_out + 1)])
    valid = (ell <= lmax_in) & (m <= lmax_in)
    return np.where(valid, m * (2 * lmax_in + 1 - m) // 2 + ell, 0), valid


class ScatterTransform:
    """Scattering covariances of fields at one band limit, with everything reusable precomputed.

    The base class takes harmonic coefficients (healpy ordering, any lmax, transferred to
    lmax = L-1); ``CARScatterTransform`` and ``HealpixScatterTransform`` take maps. The wavelet
    maps and statistics are always computed on full-sky CAR Fejer-1 geometries, one per scale,
    optionally cut to the rows covering a footprint.

    Args:
        lmax: band limit of the input; L = lmax + 1 rows in the finest CAR geometry.
        N: azimuthal band limit; 2N-1 wavelet orientations (N must be odd).
        J_min: lowest wavelet scale; the highest is log2 L.
        mask: optional boolean mask, either an enmap on the full-sky Fejer-1 geometry with L rows,
            ``enmap.fullsky_geometry(shape=(L, 2 * L), variant="fejer1")``, or a HEALPix map.
            Pixel statistics are restricted to it and, with ``cut``, every scale is restricted to
            the rows covering it.
        cut: restrict the per-scale geometries to the rows covering the mask, so that every
            synthesis costs proportionally less (rows are kept whole; pixell pads nothing).
        lmax_trunc: analyse |W_j2| only up to the band limit needed by the coarser scales.
            The analysis is the quadrature weighted adjoint on the geometry of the scale, the
            same on both backends. It is not exact for a map that is not band limited, as the
            modulus of a wavelet map never is, so some power from higher multipoles folds into
            the coefficients the second wavelet layer uses.
        nthread: threads for the pixell transforms (default: all).
        backend: "numpy" or "jax"; default from ``SOMA_SCATTERING_BACKEND``.
        jit: compile the JAX transform on first use (JAX backend only).
    """

    def __init__(
        self,
        lmax,
        N=3,
        J_min=2,
        mask=None,
        cut=False,
        lmax_trunc=True,
        nthread=None,
        backend=None,
        jit=True,
    ):
        L = lmax + 1
        shape, wcs = enmap.fullsky_geometry(shape=(L, 2 * L), variant="fejer1")
        if mask is not None and np.ndim(mask) == 1:  # HEALPix mask: rasterise onto the CAR geometry
            mask = enmap.ndmap(_healpix_to_mask(mask, shape, wcs), wcs)
        elif mask is not None and tuple(mask.shape[-2:]) != tuple(shape):
            raise ValueError(
                "mask must be an enmap on the full-sky Fejer-1 geometry or a HEALPix map"
            )
        self.shape, self.wcs = shape, wcs
        self.backend = backend or _default_backend()
        if self.backend == "jax" and not _jax_available():
            raise RuntimeError(
                "the JAX backend needs JAX and a ducc built with its XLA handlers "
                "(DUCC0_XLA_INCLUDE set to jax.ffi.include_dir(), and DUCC0_USE_CUDA "
                "for a GPU)"
            )
        self.L, self.N, self.J_min, self.J, self.ndir = L, N, J_min, _j_max(L), 2 * N - 1
        self.lmax_trunc, self.nthread = lmax_trunc, nthread
        J = self.J
        psi, G, self.orders = _directional_filters(L, N), _steering_matrix(N), _active_orders(N)
        self.ncomp = sum(1 if n == 0 else 2 for n in self.orders)
        cl = np.sqrt(4 * np.pi / (2 * np.arange(L) + 1))
        self.ainfo = curvedsky.alm_info(lmax=L - 1)
        self.Lj = {j: min(int(np.ceil(2.0 ** (j + 1))), L) for j in range(J_min, J + 1)}
        Ls = sorted(set(self.Lj.values()))
        box = None
        if mask is not None and cut:
            rows = np.where(np.asarray(mask).any(axis=1))[0]
            box = enmap.pixbox2skybox(mask.shape, mask.wcs, [[rows[0], 0], [rows[-1] + 1, 0]])[:, 0]
        self.geoms = {}
        for Lj in Ls:
            s, w = enmap.fullsky_geometry(shape=(Lj, 2 * Lj), variant="fejer1")
            # rows covering the declination range, rounded outwards, full columns
            if box is not None:
                y = enmap.sky2pix(s, w, np.array([box, [0, 0]]))[0]
                y0, y1 = (
                    int(np.clip(np.floor(y.min()), 0, s[0])),
                    int(np.clip(np.ceil(y.max()) + 1, 0, s[0])),
                )
                s, w = enmap.subgeo(s, w, pixbox=np.array([[y0, 0], [y1, s[1]]]))
            self.geoms[Lj] = (s, w)
        self.ainfos = {Lj: curvedsky.alm_info(lmax=Lj - 1) for Lj in Ls}
        self.binfos = {
            j2: curvedsky.alm_info(lmax=(self.Lj[j2 - 1] if lmax_trunc else self.Lj[j2]) - 1)
            for j2 in range(J_min + 1, J + 1)
        }
        ell = {Lj: _alm_ell(Lj - 1) for Lj in Ls}
        lmax_ins = {L - 1} | {b.lmax for b in self.binfos.values()}
        self.tidx = {
            (li, Lj): _transfer_index(li, Lj - 1) for li in lmax_ins for Lj in Ls if li != Lj - 1
        }
        fl = {
            (j, n): ((cl * psi[j, :, n])[: self.Lj[j]] * (-1) ** n)[ell[self.Lj[j]]]
            for j in self.Lj
            for n in self.orders
        }
        w, self.norm, self.mask = {}, {}, {}
        for Lj in Ls:
            s, ww = self.geoms[Lj]
            wt = curvedsky.quad_weights(s, ww)[:, None] * np.ones(s)
            self.mask[Lj] = (
                None
                if mask is None
                else enmap.project(mask.astype(np.float64), s, ww, order=0) > 0.5
            )
            if mask is not None:
                wt *= self.mask[Lj]
            w[Lj], self.norm[Lj] = wt, float(wt.sum())
        # backend specifics: array namespace, constant arrays, transforms and buffers
        self.bufs = {}
        if self.backend == "jax":
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            self.xp = jnp
            self.G, self.fl, self.w = (
                jnp.asarray(G),
                {k: jnp.asarray(v) for k, v in fl.items()},
                {k: jnp.asarray(v) for k, v in w.items()},
            )
            self.tidx = {k: (jnp.asarray(i), jnp.asarray(v)) for k, (i, v) in self.tidx.items()}
            self.rgeom = {Lj: curvedsky.RingGeometry(*self.geoms[Lj]) for Lj in Ls}
            self._fn = jax.jit(self._stats) if jit else self._stats
        else:
            self.xp, self.G, self.fl, self.w = np, G, fl, w
            self.spinbuf = {
                Lj: np.zeros((2, self.ainfos[Lj].nelem), dtype=np.complex128) for Lj in Ls
            }
            self.comps = {
                Lj: enmap.zeros((self.ncomp,) + self.geoms[Lj][0], self.geoms[Lj][1]) for Lj in Ls
            }
            self._fn = self._stats

    # ---- backend primitives ---------------------------------------------------------------

    def _buf(self, key, shape, dtype=np.float64):
        """A persistent work array for the numpy backend (allocated once per key); zeros for JAX."""
        if self.backend == "jax":
            return self.xp.zeros(shape, dtype)
        if key not in self.bufs or self.bufs[key].shape != tuple(shape):
            self.bufs[key] = np.empty(shape, dtype)
        return self.bufs[key]

    def _set(self, arr, idx, val):
        """arr[idx] = val for numpy, arr.at[idx].set(val) for JAX; returns the updated array."""
        if self.backend == "jax":
            return arr.at[idx].set(val)
        arr[idx] = val
        return arr

    def _synth(self, a, Lj, spin, k):
        """Inverse transform of a (spin 0) or the (E, 0) pair (spin > 0) onto the geometry of Lj.

        Returns the map components; for numpy they are written into rows k.. of the component
        buffer, for JAX a new array is returned.
        """
        if self.backend == "jax":
            # an explicit component axis: (a) for spin 0, the pair (E, B = 0) otherwise
            a = a[None] if spin == 0 else self.xp.stack([a, self.xp.zeros_like(a)])
            return curvedsky.alm2map(
                a, self.rgeom[Lj], spin=[spin], ainfo=self.ainfos[Lj], nthread=self.nthread
            ).arr
        n = 1 if spin == 0 else 2
        if spin > 0:
            self.spinbuf[Lj][0] = a
            a = self.spinbuf[Lj]
        return curvedsky.alm2map(
            a, self.comps[Lj][k : k + n], spin=[spin], ainfo=self.ainfos[Lj], nthread=self.nthread
        )

    def _transfer(self, a, lmax_in, Lj):
        """alm at lmax_in truncated or zero padded to the layout of band limit Lj."""
        if lmax_in == Lj - 1:
            return a
        idx, valid = self.tidx[lmax_in, Lj]
        return self.xp.where(valid, a[idx], 0)

    def analysis(self, m, j2):
        """Harmonic coefficients (layout binfos[j2]) of a real map on the geometry of scale j2."""
        wcs = self.geoms[self.Lj[j2]][1]
        m = enmap.devmap(m, wcs) if self.backend == "jax" else enmap.ndmap(m, wcs)
        return curvedsky.map2alm(
            m, ainfo=self.binfos[j2], spin=[0], method="cyl", nthread=self.nthread
        )

    # ---- the transform -----------------------------------------------------------------------

    def wavelet(self, a, lmax_in, j):
        """Directional wavelet maps (2N-1, ny, nx) of alm a (healpy layout, lmax_in) at scale j."""
        Lj = self.Lj[j]
        a = self._transfer(a, lmax_in, Lj)
        comps, k = [], 0
        for n in self.orders:
            if n == 0:
                comps.append(self._synth(a * self.fl[j, n], Lj, 0, k)[0])
                k += 1
            else:  # E = -c_l psi f, B = 0; f_n = Q - iU
                qu = self._synth(-a * self.fl[j, n], Lj, n, k)
                comps += [qu[0], -qu[1]]
                k += 2
        return self.xp.tensordot(self.G, self.xp.stack(comps), axes=(1, 0))

    def _stats(self, alm, alm2=None, timer=None):
        xp, L, J, J_min, ndir = self.xp, self.L, self.J, self.J_min, self.ndir
        alms = [alm] if alm2 is None else [alm, alm2]
        nf = len(alms)
        means = xp.stack([xp.abs(a[0]) / (2 * xp.sqrt(xp.pi)) for a in alms])
        prod = xp.real(alms[0] * xp.conj(alms[-1]))
        var = (xp.sum(prod[1:L]) + 2 * xp.sum(prod[L:])) / (4 * xp.pi)
        W = [{j: self.wavelet(a, L - 1, j) for j in self.Lj} for a in alms]
        _tick(timer, f"first wavelet layer, {len(self.Lj)} scales")
        S1, P00 = [[] for _ in alms], []
        N2 = [
            {
                j1: self._buf((k, "N2", j1), (ndir, J - j1, ndir) + self.geoms[self.Lj[j1]][0])
                for j1 in range(J_min, J)
            }
            for k in range(nf)
        ]
        for j2 in self.Lj:
            Lj2 = self.Lj[j2]
            w, norm = self.w[Lj2], self.norm[Lj2]
            P00.append(xp.einsum("gtp,gtp,tp->g", W[0][j2], W[-1][j2], w) / norm)
            for k in range(nf):
                absW = xp.abs(W[k][j2])
                S1[k].append(xp.einsum("gtp,tp->g", absW, w) / norm)
                if j2 == J_min:
                    continue
                for g2 in range(ndir):
                    M = self.analysis(absW[g2], j2)
                    for j1 in range(J_min, j2):
                        N2[k][j1] = self._set(
                            N2[k][j1],
                            (slice(None), j2 - j1 - 1, g2),
                            self.wavelet(M, self.binfos[j2].lmax, j1),
                        )
        _tick(timer, "S1, P00 and the second wavelet layer")
        C01, C11 = [], []
        for j1 in range(J_min, J):
            w, norm = self.w[self.Lj[j1]], self.norm[self.Lj[j1]]
            A1, A2, W1 = N2[0][j1], N2[-1][j1], W[0][j1]
            nj2 = A1.shape[1]
            c01, c11 = [], []
            for g1 in range(ndir):
                Ag = A1[g1].reshape(nj2 * ndir, -1)
                Bg = xp.multiply(A2[g1], w).reshape(nj2 * ndir, -1)  # second field, weighted
                c01.append(Bg @ W1[g1].ravel())
                c11.append(Ag @ Bg.T)
            c01 = xp.stack(c01, axis=-1).reshape(nj2, ndir, ndir)  # [j2, gamma2, gamma1]
            c11 = xp.stack(c11).reshape(ndir, nj2, ndir, nj2, ndir).transpose(1, 3, 2, 4, 0)
            C01.append(c01.ravel() * (4 * xp.pi / norm))
            C11.append(c11.ravel() * (4 * xp.pi / norm))
        _tick(timer, "covariances")
        S1 = xp.stack([xp.concatenate(s) for s in S1])
        return (
            means[0] if alm2 is None else means,
            var,
            S1[0] if alm2 is None else S1,
            xp.concatenate(P00),
            xp.concatenate(C01),
            xp.concatenate(C11),
        )

    def _prepare(self, alm):
        # JAX arrays and tracers (the arrays with an .at updater) pass through untouched, so
        # that jit, grad and vmap see them; anything else becomes numpy at lmax = L-1
        if not hasattr(alm, "at"):
            alm = np.asarray(alm, dtype=np.complex128)
            if alm.size != self.ainfo.nelem:
                alm = curvedsky.transfer_alm(curvedsky.alm_info(nalm=alm.size), alm, self.ainfo)
        return self.xp.asarray(alm, dtype=self.xp.complex128)

    def __call__(self, alm, alm2=None, timer=None):
        """Return (mean, var, S1, P00, C01, C11) for alm in healpy ordering (any lmax).

        With a second field ``alm2`` the cross statistics are returned (see ``from_alm``).
        ``timer`` is an optional object with a ``tick(msg)`` method, called as each stage
        finishes (numpy backend only: the JAX backend runs as one compiled function).
        """
        return self.from_alm(
            self._prepare(alm), None if alm2 is None else self._prepare(alm2), timer
        )

    def from_alm(self, alm, alm2=None, timer=None):
        """Statistics from healpy ordered alm with exactly lmax = L-1 (no layout transfer).

        Available on every subclass, so a map based object can also be fed alm directly.

        With a second field ``alm2`` (same layout) the cross statistics of the pair (f, g) are
        returned in the same flattened layout: ``P00`` = <W f . W g>, ``C01`` = <W f . W|W g|>
        (the first field enters linearly, the second through the modulus), ``C11`` =
        <W|W f| . W|W g|>, and ``var`` the cross variance from the harmonic coefficients;
        ``mean`` and ``S1``, which are single-field quantities, are returned for both fields
        stacked along a leading axis of length 2. With ``alm2`` omitted or identical to ``alm``
        the auto statistics of s2scat are recovered.
        """
        if self.backend == "jax":
            return self._fn(alm) if alm2 is None else self._fn(alm, alm2)
        return self._stats(alm, alm2, timer)


class CARScatterTransform(ScatterTransform):
    """Scattering covariances of enmaps on the full-sky CAR Fejer-1 geometry (shape, wcs).

    Same arguments as ``ScatterTransform`` after ``shape, wcs`` (which fix L = shape[-2]).
    """

    def __init__(self, shape, wcs, **kwargs):
        if shape[-1] != 2 * shape[-2]:
            raise ValueError(
                'shape, wcs must be enmap.fullsky_geometry(shape=(L, 2 * L), variant="fejer1")'
            )
        super().__init__(shape[-2] - 1, **kwargs)

    def __call__(self, imap, imap2=None, timer=None):
        """Return the statistics of a real enmap on this geometry (cross statistics with imap2)."""
        alms = [
            curvedsky.map2alm(m, ainfo=self.ainfo, nthread=self.nthread)
            for m in [imap, imap2]
            if m is not None
        ]
        _tick(timer, "map2alm")
        return self.from_alm(*alms, timer=timer)


class HealpixScatterTransform(ScatterTransform):
    """Scattering covariances of HEALPix maps (ring ordering), analysed with ``healpy.map2alm``.

    Args:
        lmax: band limit of the analysis.
        niter: iterations of ``healpy.map2alm`` (0: plain quadrature, accurate to 1e-6 in the
            statistics for nside >= L/2; 3 gives 1e-11).
        Other arguments as for ``ScatterTransform``; ``mask`` may be a HEALPix map.
    """

    def __init__(self, lmax, niter=0, **kwargs):
        super().__init__(lmax, **kwargs)
        self.niter = niter

    def __call__(self, hmap, hmap2=None, timer=None):
        """Return the statistics of a real HEALPix map (cross statistics with hmap2)."""
        import healpy as hp

        alms = [
            hp.map2alm(np.asarray(m, dtype=np.float64), lmax=self.L - 1, iter=self.niter)
            for m in [hmap, hmap2]
            if m is not None
        ]
        _tick(timer, "map2alm")
        return self.from_alm(*alms, timer=timer)
