"""Command line front end for :func:`somapy.io.plot_footprints`.

    soma-footprints act_ivar.fits so_ivar.fits -o footprints.gif

Installed as the ``soma-footprints`` console script, so it is on PATH after
``pip install somapy``.
"""

import argparse

from somapy import __version__
from somapy.io import plot_footprints


def build_parser():
    p = argparse.ArgumentParser(
        prog="soma-footprints",
        description="Show the sky footprint of pixell ivar maps on a rotating globe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = p.add_argument
    add("ivars", nargs="+", metavar="IVAR", help="ivar map(s), as FITS files")
    add(
        "-o",
        "--output",
        default="footprints.gif",
        help=".gif for the animation, .png/.pdf/.svg for a still",
    )
    add("-l", "--labels", help="comma-separated names, one per map (default: the file names)")
    add("-c", "--colors", help="comma-separated matplotlib colours, one per map")
    add("-t", "--threshold", type=float, default=0.0, help="a pixel is observed where ivar > this")
    add("--title", help="figure title")

    g = p.add_argument_group("animation")
    g.add_argument("--nframes", type=int, default=180, help="frames per turn; 1 gives a still")
    g.add_argument("--fps", type=int, default=15, help="playback rate")
    g.add_argument(
        "--ra0", type=float, default=0.0, help="RA at the centre of the first frame, degrees"
    )
    g.add_argument(
        "--dec0", type=float, default=-20.0, help="declination the view is centred on, degrees"
    )
    g.add_argument(
        "--wobble",
        type=float,
        default=20.0,
        help="north-south nod over the turn, degrees; 0 for a pure spin",
    )

    g = p.add_argument_group("appearance")
    g.add_argument("--size", type=int, default=720, help="globe size in pixels")
    g.add_argument(
        "--res", type=float, default=0.2, help="internal all-sky grid resolution, degrees"
    )
    g.add_argument("--alpha", type=float, default=0.7, help="opacity of the fills")
    g.add_argument(
        "--grid", type=float, default=30.0, help="graticule spacing in degrees; 0 disables it"
    )
    g.add_argument(
        "--ra-units", choices=("hours", "deg"), default="hours", help="units for the RA labels"
    )
    g.add_argument(
        "--ra-sign",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="-1 puts RA increasing to the left, as seen from Earth",
    )
    g.add_argument("--galactic-plane", action="store_true", help="overlay the Galactic plane")
    g.add_argument(
        "--backdrop",
        default="auto",
        help="sky map to paint the globe with; 'auto' for the bundled "
        "Planck dust map, 'none' for a plain globe",
    )

    p.add_argument("--version", action="version", version=f"soma {__version__}")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    split = lambda s: [x.strip() for x in s.split(",")] if s else None  # noqa: E731
    areas = plot_footprints(
        a.ivars,
        labels=split(a.labels),
        output=a.output,
        threshold=a.threshold,
        colors=split(a.colors),
        nframes=a.nframes,
        fps=a.fps,
        dec0=a.dec0,
        wobble=a.wobble,
        ra0=a.ra0,
        grid=a.grid,
        ra_units=a.ra_units,
        size=a.size,
        res=a.res,
        galactic_plane=a.galactic_plane,
        title=a.title,
        ra_sign=a.ra_sign,
        alpha=a.alpha,
        backdrop=None if a.backdrop.lower() == "none" else a.backdrop,
    )
    for label, area in areas.items():
        print(f"{label}: {area:,.0f} deg^2")
    if a.output:
        print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":  # python -m somapy.scripts.footprints
    raise SystemExit(main())
