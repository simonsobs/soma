"""Reading survey maps and showing what they cover.

    from somapy.io import plot_footprints
    plot_footprints({"ACT DR6": "act_ivar.fits", "SO SAT": "so_ivar.fits"})

Draws the sky footprint of a set of pixell ivar maps on a rotating globe.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from astropy.visualization import (
    AsymmetricPercentileInterval,
    ImageNormalize,
    PowerStretch,
)
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.textpath import TextPath
from pixell import coordinates, enmap, utils

__all__ = ["plot_footprints"]

# Fixed categorical order, tuned for the dark surface below. Only the first
# three stay reliably distinguishable when regions can touch, which is what the
# warning past three is about.
COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SURFACE, INK, MUTED, GLOBE = "#1a1a19", "#ffffff", "#c3c2b7", "#3a3a37"
# Dim -> bright ramp the sky texture is painted with. Any matplotlib colormap
# works here; this one is deliberately neutral so it stays behind the fills.
BACKDROP_CMAP = LinearSegmentedColormap.from_list("sky", ["#1b1b1a", "#b3b0a3"])
BACKDROP_GAMMA = 2.0  # midtone bend, see _backdrop()
BACKDROP_ENV = "SOMA_SKY_BACKDROP"  # path to the prepared sky map
BACKDROP_FILE = "planck_dust_equ.fits"  # shipped as soma package data
_D2R = np.pi / 180
_LIMIT = 1.24  # axes half-width, in sphere radii
_AMBIENT = 0.66  # darkest a fill goes at the limb
_SHADE_STEPS = np.linspace(_AMBIENT, 1.0, 5)  # sampled for the GIF palette


def plot_footprints(
    ivars,
    labels=None,
    output="footprints.gif",
    *,
    threshold=0.0,
    colors=None,
    nframes=180,
    fps=15,
    dec0=-20.0,
    wobble=20.0,
    ra0=0.0,
    grid=30.0,
    ra_units="hours",
    size=720,
    res=0.2,
    galactic_plane=False,
    title=None,
    ra_sign=-1,
    backdrop="auto",
    alpha=0.7,
):
    """Draw the footprints of a set of ivar maps on a rotating sphere.

    Parameters
    ----------
    ivars : enmap, str, sequence or dict
        Inverse-variance maps, in memory or as paths to FITS files; a dict maps
        label -> map. A pixel is inside the footprint where ``ivar > threshold``
        in any component (leading axes such as TQU are collapsed); NaNs never
        count. Pre-downgrade them if memory is tight.
    labels : sequence of str, optional
        Footprint names. Defaults to dict keys, file names, or "map N".
    output : str or None
        ``.gif`` writes the rotating animation, ``.png``/``.pdf``/``.svg`` a
        single still. ``None`` renders without writing.
    threshold : float
        Footprint cut on the ivar value.
    colors : sequence of str, optional
        One hex colour per map; defaults to ``COLORS`` in order.
    nframes, fps : int
        Frames per full turn and playback rate; together they set how smooth
        and how fast the turn is (the default is 2 degrees per frame, taking
        12.6 s. ``nframes=1`` gives a still.
    dec0 : float
        Declination the view is centred on; how far the globe is tilted. The
        default sits mid-way through dec -65 to +25, the band a ground-based
        survey from the south typically reaches.
    wobble : float
        Amplitude, in degrees, of a north-south nod over the turn, so the south
        pole comes into view rather than only the band around ``dec0``. Set
        ``wobble=0`` for a pure rotation.
    ra0 : float
        RA at the centre of the first frame, in degrees.
    grid : float
        Graticule spacing in degrees; 0 disables it.
    ra_units : {"hours", "deg"}
        Units for the RA labels.
    size : int
        Pixel size of the globe.
    res : float
        Resolution of the internal all-sky grid, in degrees.
    galactic_plane : bool
        Overlay the Galactic plane as a dashed curve.
    title : str, optional
        Figure title. A lone footprint gets its name and area by default, since
        it carries no legend.
    ra_sign : {-1, 1}
        ``-1`` puts RA increasing to the left (the sky as seen from Earth);
        ``+1`` shows the celestial sphere as if from outside.
    backdrop : "auto", str, enmap or None
        Sky map the globe is painted with, as a dim grey texture under the
        footprints; by default the prepared Planck dust map, if one has been
        provided (see the docs), and a plain globe otherwise. Pass a path or
        an equatorial enmap to use another, or ``None`` to force plain.
    alpha : float
        Opacity of the footprint fills, so the sky beneath still reads through
        them. ``1`` paints them solid.

    Returns
    -------
    dict
        ``{label: sky area in square degrees}``.
    """
    items = _sources(ivars, labels)
    if not items:
        raise ValueError("no ivar maps supplied")
    colors = list(colors) if colors is not None else COLORS[: len(items)]
    if len(colors) < len(items):
        raise ValueError(f"got {len(colors)} colours for {len(items)} maps")
    if len(items) > 3:
        warnings.warn(
            f"{len(items)} footprints on one globe: past three, colours "
            "alone stop separating regions that touch; names are "
            "written on the footprints too",
            RuntimeWarning,
            stacklevel=2,
        )

    # One shared all-sky grid, so maps of any WCS can be drawn together and the
    # per-frame cost no longer depends on the input map sizes.
    gshape, gwcs = enmap.fullsky_geometry(res=res * utils.degree, proj="car")
    row_area = np.asarray(enmap.pixsizemap(gshape, gwcs, separable=True))[:, :1] / utils.degree**2
    sky = _backdrop(backdrop, gshape, gwcs)
    masks, areas, anchors = [], {}, []
    for label, src in items:
        m = _footprint(src, threshold, gshape, gwcs, res)
        if not m.any():
            warnings.warn(
                f"footprint {label!r} is empty at threshold {threshold}",
                RuntimeWarning,
                stacklevel=2,
            )
        masks.append(m.reshape(-1))
        areas[label] = float((m * row_area).sum())
        anchors.append(_label_anchor(m, gshape, gwcs))
    # Paint the largest first so small footprints are never buried.
    order = sorted(range(len(items)), key=lambda k: -areas[items[k][0]])
    if title is None and len(items) == 1:  # one footprint needs no legend
        title = f"{items[0][0]} — {areas[items[0][0]]:,.0f} deg²"

    # --- screen geometry. Spinning in RA is only an index shift along the
    # grid's x axis, so it is solved once; tilting in dec is not, so it is
    # re-solved per frame, and only when the globe actually wobbles.
    ss = 2  # supersampling, for a clean limb
    S = size * ss
    c = np.linspace(-1, 1, S + 1)
    x, y = np.meshgrid(0.5 * (c[1:] + c[:-1]), 0.5 * (c[1:] + c[:-1]))
    rho2 = x * x + y * y
    inside = rho2 <= 1
    z = np.sqrt(np.clip(1 - rho2, 0, 1))
    nx, cdelt = gshape[-1], float(gwcs.wcs.cdelt[0])
    idx_in = np.nonzero(inside.reshape(-1))[0]
    # Diffuse shading from an upper-left light, so the disc reads as a ball.
    light = np.array([-0.35, 0.42, 0.84]) / np.linalg.norm([-0.35, 0.42, 0.84])
    shade = (
        _AMBIENT
        + (1 - _AMBIENT) * np.clip(x * light[0] + y * light[1] + z * light[2], 0, 1) ** 0.75
    ).astype(np.float32)
    palette = np.array(
        [to_rgb(GLOBE)] + [to_rgb(c) for c in colors[: len(items)]], dtype=np.float32
    )

    # --- figure
    still = nframes <= 1 or (output and os.path.splitext(output)[1].lower() != ".gif")
    fs = max(7.0, size / 78)
    legend, lcols, lrows = _legend(items, areas, colors, galactic_plane, alpha, size / 100, fs)
    fig = Figure(
        figsize=(size / 100, (size + (46 if title else 10)) / 100 + 0.34 * lrows),
        dpi=100,
        facecolor=SURFACE,
    )
    FigureCanvasAgg(fig)
    h = fig.get_figheight()
    ax = fig.add_axes([0, 0.34 * lrows / h, 1, size / 100 / h])
    if title:
        fig.text(
            0.5,
            1 - 0.23 / h,
            title,
            ha="center",
            va="center",
            color=INK,
            fontsize=max(11, size / 52),
        )
    if lrows:
        fig.legend(
            handles=legend,
            loc="lower center",
            frameon=False,
            fontsize=fs,
            ncol=lcols,
            labelcolor=INK,
            handlelength=1.4,
            handleheight=0.9,
            borderaxespad=0.35,
        )

    regions = [(lab, col, anc) for (lab, _), col, anc in zip(items, colors, anchors, strict=False)]
    disc = inside.astype(np.float32).reshape(size, ss, size, ss).mean((1, 3))
    phase = np.zeros(1) if still else np.linspace(0, 1, nframes, endpoint=False)
    views = ra0 + 360 * phase
    # A full sine over the turn, so the loop closes and the still is untilted.
    tilts = dec0 + wobble * np.sin(2 * np.pi * phase)

    frames, grid_index = [], None
    for view, tilt in zip(views, tilts, strict=True):
        if grid_index is None or wobble:
            grid_index = _screen_to_grid(x, y, z, inside, tilt, ra_sign, gshape, gwcs)
        iy, ix0 = grid_index
        ids = np.full(S * S, -1, dtype=np.int16)
        ix = np.rint(ix0 + view / cdelt).astype(np.int64) % nx
        for k in order:
            hit = masks[k][iy + ix]
            ids[idx_in[hit]] = k
        _draw(
            ax,
            ids.reshape(S, S),
            palette,
            shade,
            disc,
            ss,
            size,
            view,
            tilt,
            ra_sign,
            grid,
            ra_units,
            fs,
            regions,
            galactic_plane,
            alpha=alpha,
            tex=None if sky is None else sky[iy + ix],
            idx_in=idx_in,
        )
        if not still:
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    if output:
        if still:
            fig.savefig(output, facecolor=SURFACE)
        else:
            _write_gif(frames, output, fps, keep=colors[: len(items)] + [GLOBE])
    return areas


# ---------------------------------------------------------------- ingest


def _sources(ivars, labels):
    """Accept an enmap, a filename, a sequence of either, or {label: map}."""
    if isinstance(ivars, dict):
        items = list(ivars.items())
    elif isinstance(ivars, (str, os.PathLike)) or hasattr(ivars, "wcs"):
        items = [(None, ivars)]
    else:
        items = [(None, m) for m in ivars]
    if labels is not None:
        labels = list(labels)
        if len(labels) != len(items):
            raise ValueError(f"got {len(labels)} labels for {len(items)} maps")
        items = [(lab, src) for lab, (_, src) in zip(labels, items, strict=True)]
    out = []
    for i, (label, src) in enumerate(items):
        if label is None:
            label = (
                os.path.basename(os.fspath(src)).split(".")[0]
                if isinstance(src, (str, os.PathLike))
                else f"map {i + 1}"
            )
        out.append((str(label), src))
    return out


def _footprint(src, threshold, gshape, gwcs, res):
    """Sample one ivar map's coverage onto the shared all-sky grid."""
    imap = enmap.read_map(src) if isinstance(src, (str, os.PathLike)) else src
    with np.errstate(invalid="ignore"):
        cover = np.asarray(imap) > threshold
    if cover.ndim > 2:
        cover = cover.reshape((-1,) + cover.shape[-2:]).any(0)
    cover = enmap.enmap(cover.astype(np.float32), imap.wcs, copy=False)
    factor = max(1, int(res / abs(float(imap.wcs.wcs.cdelt[1]))))
    # A map that goes all the way round in RA has no real edge at its seam, but
    # both steps below treat one as if it did: downgrade drops the columns left
    # over by a width that is not a whole number of blocks, and the projection
    # has nothing to interpolate with beyond the last column, so it falls back
    # on cval. Either leaves a meridian of empty pixels at the seam. Carrying
    # the far side's columns across it first fixes both; a map that stops short
    # of a full turn picks up zeros there instead, as it should.
    cover = _wrap_pad(cover, 2 * factor, 2 * factor + (-cover.shape[-1] % factor))
    if factor > 1:
        cover = enmap.downgrade(cover, factor, op=np.max)  # keeps thin coverage
    proj = enmap.project(cover, gshape, gwcs, order=0, border="constant", cval=0.0)
    return np.asarray(proj) > 0.5


def _wrap_pad(m, left, right):
    """Widen a map in RA, taking the extra columns from the opposite edge where
    it wraps round the sky and filling with zeros where it does not."""
    return enmap.extract_pixbox(m, [[0, -left], [m.shape[-2], m.shape[-1] + right]], cval=0.0)


def _bundled_backdrop():
    """Path of the sky map shipped with the code."""
    if __package__:
        try:
            from importlib.resources import files

            path = files(__package__) / BACKDROP_FILE
            if path.is_file():
                return str(path)
        except (ImportError, TypeError, ModuleNotFoundError, FileNotFoundError):
            pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), BACKDROP_FILE)


def _backdrop(src, gshape, gwcs):
    """The sky texture the globe is painted with, on the working grid.

    Returns values normalised to 0..1 (log stretch, percentile clipped), flat,
    or None for a plain globe. ``src="auto"`` uses the prepared Planck dust map
    if one has been provided, and falls back to plain if not.
    """
    if src is None:
        return None
    if isinstance(src, str) and src == "auto":
        src = os.environ.get(BACKDROP_ENV) or _bundled_backdrop()
        if not src or not os.path.exists(src):
            return None
    imap = enmap.read_map(src) if isinstance(src, (str, os.PathLike)) else src
    imap = enmap.enmap(
        np.asarray(imap, dtype=np.float32).reshape((-1,) + imap.shape[-2:])[0], imap.wcs, copy=False
    )
    v = np.asarray(enmap.project(imap, gshape, gwcs, order=1, border="wrap"))
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    pos = v[v > 0]
    if not pos.size:
        return None
    v = np.log10(np.maximum(v, np.percentile(pos, 1)))  # dust spans decades
    # Percentiles rather than fixed limits, so the stretch adapts to whatever map
    # is supplied; the power bends the midtones down, since a straight ramp leaves
    # the sky a flat mid-grey with the Galactic plane barely brighter. clip=True
    # is load-bearing; without it the stretch squares below-range values back
    # up and the darkest sky comes out bright.
    norm = ImageNormalize(
        v,
        interval=AsymmetricPercentileInterval(5, 99.5),
        stretch=PowerStretch(BACKDROP_GAMMA),
        clip=True,
    )
    return np.ma.filled(norm(v), 0).astype(np.float32).reshape(-1)


def _label_anchor(mask, gshape, gwcs):
    """The RA/dec a footprint's name is pinned to: the point of the footprint
    furthest from its own edge.

    Solved once, on the sky rather than on the screen, so the label sits at a
    fixed place on the sphere and turns with it instead of sliding about inside
    the footprint from frame to frame.
    """
    if not mask.any():
        return None
    try:
        from scipy import ndimage  # a pixell dependency
    except ImportError:  # pragma: no cover
        return None
    nx = mask.shape[1]
    # Wrap in RA so a footprint crossing RA=0 is not cut in two; pad in dec so
    # one running up to the pole does not claim clearance beyond it.
    wide = np.pad(np.concatenate([mask] * 3, 1), ((1, 1), (0, 0)))
    dist = ndimage.distance_transform_edt(wide)[1:-1, nx : 2 * nx]
    ys, xs = np.nonzero(dist >= dist.max() - 0.5)  # a band ties along its
    j = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)  # spine: take the
    cd, cp, cv = gwcs.wcs.cdelt, gwcs.wcs.crpix, gwcs.wcs.crval  # middle one of them
    return (float(cv[0] + (xs[j] + 1 - cp[0]) * cd[0]), float(cv[1] + (ys[j] + 1 - cp[1]) * cd[1]))


def _text_fits(seen, x, y, label, fs, dpi):
    """Is the footprint solid enough around ``(x, y)`` to hold the text?"""
    n = seen.shape[0]
    px = dpi / 72 * _LIMIT  # points -> raster pixels
    w, h = _text_pt(label, fs)  # + a little for the halo it wears
    hw, hh = 0.5 * w * px + 2, 0.5 * max(h, 0.9 * fs) * px + 2
    ix, iy = (x + 1) / 2 * n - 0.5, (y + 1) / 2 * n - 0.5
    if min(ix - hw, iy - hh) < 0 or max(ix + hw, iy + hh) >= n - 1:
        return False  # would run off the globe
    box = seen[int(iy - hh) : int(iy + hh) + 1, int(ix - hw) : int(ix + hw) + 1]
    return box.mean() > 0.9


# ---------------------------------------------------------------- drawing


def _screen_to_grid(x, y, z, inside, dec0, ra_sign, gshape, gwcs):
    """Screen pixel -> flat index into the all-sky grid, for one tilt.

    Inverse orthographic, then straight into pixel coordinates: the grid is
    CAR, so both axes are linear in the sky angles and no WCS call is needed.
    RA comes out relative to the view centre, which is what lets the rotation
    stay a plain index shift.
    """
    sin0, cos0 = np.sin(dec0 * _D2R), np.cos(dec0 * _D2R)
    xi, yi, zi = x[inside], y[inside], z[inside]
    dec = np.degrees(np.arcsin(np.clip(zi * sin0 + yi * cos0, -1, 1)))
    ra = np.degrees(np.arctan2(ra_sign * xi, zi * cos0 - yi * sin0))
    cd, cp, cv = gwcs.wcs.cdelt, gwcs.wcs.crpix, gwcs.wcs.crval
    iy = np.clip(np.rint(cp[1] - 1 + (dec - cv[1]) / cd[1]), 0, gshape[-2] - 1)
    return iy.astype(np.int64) * gshape[-1], cp[0] - 1 + (ra - cv[0]) / cd[0]


def _forward(ra, dec, ra0, dec0, ra_sign):
    """Sky -> screen; the point is on the near side where the third value > 0."""
    dl = (np.asarray(ra, float) - ra0) * _D2R
    d, d0 = np.asarray(dec, float) * _D2R, dec0 * _D2R
    return (
        ra_sign * np.cos(d) * np.sin(dl),
        np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(dl),
        np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(dl),
    )


def _draw(
    ax,
    ids,
    palette,
    shade,
    disc,
    ss,
    size,
    view,
    dec0,
    ra_sign,
    grid,
    ra_units,
    fs,
    regions,
    galactic_plane,
    alpha=1.0,
    tex=None,
    idx_in=None,
):
    ax.clear()
    ax.set_xlim(-_LIMIT, _LIMIT)
    ax.set_ylim(-_LIMIT, _LIMIT)
    ax.set_aspect("equal")
    ax.axis("off")

    gapped = ids
    if len(regions) > 1:
        try:  # a thin gap, not an outline, where two fills meet
            from scipy import ndimage

            w = 2 * ss + 1
            gapped = np.where(
                ndimage.maximum_filter(ids, w, mode="nearest")
                != ndimage.minimum_filter(ids, w, mode="nearest"),
                -1,
                ids,
            )
        except ImportError:  # pragma: no cover
            pass
    rgb = np.empty(gapped.shape + (3,), np.float32)
    rgb[:] = palette[0]  # bare sky, then the map on it
    if tex is not None:
        rgb.reshape(-1, 3)[idx_in] = BACKDROP_CMAP(tex)[:, :3]
    covered = gapped >= 0  # fills go on translucent, so the
    rgb[covered] = (
        alpha * palette[gapped[covered] + 1]  # sky reads
        + (1 - alpha) * rgb[covered]
    )  # through them
    rgb = (rgb * shade[..., None]).reshape(size, ss, size, ss, 3)
    ax.imshow(
        np.dstack([rgb.mean((1, 3)), disc]),
        extent=(-1, 1, -1, 1),
        origin="lower",
        interpolation="antialiased",
        zorder=1,
    )

    stroke = [patheffects.withStroke(linewidth=2.4, foreground=SURFACE, alpha=0.8)]
    if grid:
        for ra in np.arange(0, 360, grid):  # meridians
            d = np.linspace(-90, 90, 181)
            _curve(ax, np.full_like(d, ra), d, view, dec0, ra_sign)
        for dec in np.arange(-90 + grid, 89.99, grid):  # parallels
            r = np.linspace(0, 360, 721)
            _curve(ax, r, np.full_like(r, dec), view, dec0, ra_sign)
        _grid_labels(ax, view, dec0, ra_sign, grid, ra_units, fs, stroke)
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=MUTED, lw=0.8, alpha=0.45, zorder=4))

    if galactic_plane:
        gl = np.linspace(0, 360, 721)
        eq = (
            coordinates.transform("gal", "equ", np.array([gl, gl * 0]) * utils.degree)
            / utils.degree
        )
        _curve(
            ax,
            eq[0],
            eq[1],
            view,
            dec0,
            ra_sign,
            color=MUTED,
            lw=1.2,
            ls=(0, (5, 3)),
            alpha=0.75,
            zorder=5,
        )

    for k, (label, col, anchor) in enumerate(regions):
        if anchor is None:
            continue
        sx, sy, near = _forward(anchor[0], anchor[1], view, dec0, ra_sign)
        if near <= 0.15:  # anchor is round the far side;
            continue  # _text_fits judges the rest
        if not _text_fits(ids[::ss, ::ss] == k, sx, sy, label, fs, ax.figure.dpi):
            continue
        ax.text(
            sx,
            sy,
            label,
            color=_ink_on(col),
            fontsize=fs,
            ha="center",
            va="center",
            zorder=7,
            path_effects=[patheffects.withStroke(linewidth=1.8, foreground=col)],
        )


def _curve(ax, ra, dec, view, dec0, ra_sign, color=MUTED, lw=0.6, ls="-", alpha=0.30, zorder=3):
    x, y, near = _forward(ra, dec, view, dec0, ra_sign)
    hidden = near <= 0
    ax.plot(
        np.where(hidden, np.nan, x),
        np.where(hidden, np.nan, y),
        color=color,
        lw=lw,
        ls=ls,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
    )


def _grid_labels(ax, view, dec0, ra_sign, grid, ra_units, fs, stroke):
    """RA along the equator, dec outside the limb."""
    ras = np.arange(0, 360, grid)
    x, y, near = _forward(ras, ras * 0, view, dec0, ra_sign)
    for ra, xi, yi, ni in zip(ras, x, y, near, strict=True):
        if ni <= 0.35 or abs(xi) > 0.82:  # foreshortened, or in the dec strip
            continue
        text = f"{ra / 15:g}h" if ra_units == "hours" else f"{ra:g}°"
        ax.text(
            xi,
            yi + 0.035,
            text,
            color=MUTED,
            fontsize=fs,
            ha="center",
            va="bottom",
            zorder=6,
            path_effects=stroke,
        )
    for dec in np.arange(-90 + grid, 89.99, grid):
        cos_dl = -np.tan(dec0 * _D2R) * np.tan(dec * _D2R)
        if not np.isfinite(cos_dl) or abs(cos_dl) > 1:
            # A parallel that never meets the limb; the globe is tilted far
            # enough that it closes around the pole. Label it on the meridian
            # through the middle instead of losing it.
            xi, yi, near = _forward(view, dec, view, dec0, ra_sign)
            if near > 0:
                ax.text(
                    xi,
                    yi,
                    f"{dec:+g}°".replace("+0", "0"),
                    color=MUTED,
                    fontsize=fs,
                    ha="center",
                    va="bottom",
                    zorder=6,
                    path_effects=stroke,
                )
            continue
        dl = np.degrees(np.arccos(cos_dl))
        for s in (1, -1):  # take the crossing on the left
            xi, yi, _ = _forward(view + s * dl, dec, view, dec0, ra_sign)
            if xi <= 0:
                r = np.hypot(xi, yi) or 1
                ax.text(
                    1.07 * xi / r,
                    1.07 * yi / r,
                    f"{dec:+g}°".replace("+0", "0"),
                    color=MUTED,
                    fontsize=fs,
                    ha="right",
                    va="center",
                    zorder=6,
                )
                break


def _legend(items, areas, colors, galactic_plane, alpha, width_in, fs):
    """Legend entries, plus the column and row counts that keep them inside the
    figure.

    Swatches are the series colour over a typical patch of sky at the fill
    opacity, so the key matches the globe rather than showing the raw hue.
    """
    if len(items) < 2 and not galactic_plane:
        return [], 1, 0
    sky = np.array(BACKDROP_CMAP(0.25)[:3])
    handles = [
        Patch(
            facecolor=tuple(alpha * np.array(to_rgb(c)) + (1 - alpha) * sky),
            edgecolor="none",
            label=f"{lab} — {areas[lab]:,.0f} deg²",
        )
        for (lab, _), c in zip(items, colors, strict=False)
    ]  # may over-supply
    if galactic_plane:
        handles.append(
            Line2D([0], [0], color=MUTED, lw=1.2, ls=(0, (5, 3)), label="Galactic plane")
        )
    entry = (max(_text_pt(h.get_label(), fs)[0] for h in handles) + 1.6 * fs + 14) / 72
    fits = max(1, min(len(handles), int(width_in / entry)))
    nrow = int(np.ceil(len(handles) / fits))  # then even the rows out, so four
    return handles, int(np.ceil(len(handles) / nrow)), nrow  # entries go 2 + 2


def _write_gif(frames, path, fps, keep=()):
    """One shared palette for every frame.

    Median cut allocates by pixel count in the reference frame, so a footprint
    that is small there, or round the far side, would otherwise lose its colour
    and come back as grey when it rotates into view.
    """
    from PIL import Image

    ramp = [tuple(int(round(255 * v * s)) for v in to_rgb(c)) for c in keep for s in _SHADE_STEPS]
    base = Image.fromarray(frames[0]).quantize(
        colors=256 - len(ramp), method=Image.Quantize.MEDIANCUT
    )
    pal = base.getpalette()[: 3 * (256 - len(ramp))] + [v for c in ramp for v in c]
    base.putpalette(pal)
    q = [Image.fromarray(f).quantize(palette=base, dither=Image.Dither.NONE) for f in frames]
    # GIF stores frame delays in centiseconds, so round there rather than let
    # the writer truncate and play the turn faster than asked for.
    q[0].save(
        path,
        save_all=True,
        append_images=q[1:],
        loop=0,
        disposal=2,
        duration=max(20, 10 * int(round(100 / fps))),
        optimize=True,
    )


def _text_pt(s, fs):
    """True width and height of a string at font size ``fs``, in points."""
    e = TextPath((0, 0), s, size=fs).get_extents()
    return e.width, e.height


def _ink_on(fill):
    """Label ink chosen from the fill's luminance, so it always clears contrast."""
    c = np.array(to_rgb(fill))
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return "#101010" if lin @ [0.2126, 0.7152, 0.0722] > 0.42 else "#ffffff"
