"""pytest suite for somapy.io."""

import numpy as np
import pytest
from PIL import Image
from pixell import enmap, utils

from somapy.io import plot_footprints


def strip(dec_range, ra_range=(-180, 180), res=20.0, value=1.0):
    """A rectangular ivar patch, sharp-edged so its area is analytic."""
    box = np.array([[dec_range[0], ra_range[0]], [dec_range[1], ra_range[1]]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj="car")
    return enmap.full(shape, wcs, value)


def exact_area(dec_range, ra_range=(-180, 180)):
    dlo, dhi = np.sin(np.radians(dec_range))
    return (dhi - dlo) * abs(ra_range[1] - ra_range[0]) * np.degrees(1.0)


def test_area_matches_analytic(tmp_path):
    areas = plot_footprints(
        strip((-30, 10), (-60, 60)), labels=["patch"], output=str(tmp_path / "a.png"), nframes=1
    )
    assert areas["patch"] == pytest.approx(exact_area((-30, 10), (-60, 60)), rel=0.02)


def test_threshold_and_component_axes(tmp_path):
    imap = strip((-20, 20), (-30, 30), value=0.5)
    tqu = enmap.enmap(np.stack([imap * 0, imap, imap * 0]), imap.wcs)  # only Q observed
    out = str(tmp_path / "a.png")
    assert plot_footprints(tqu, labels=["tqu"], output=out, nframes=1)["tqu"] > 0
    # a cut above the map value leaves nothing
    with pytest.warns(RuntimeWarning, match="empty"):
        areas = plot_footprints(tqu, labels=["tqu"], output=out, nframes=1, threshold=0.9)
    assert areas["tqu"] == 0.0


def test_nan_is_not_coverage(tmp_path):
    imap = strip((-20, 20), (-30, 30))
    imap[: imap.shape[0] // 2] = np.nan
    areas = plot_footprints(imap, labels=["nan"], output=str(tmp_path / "a.png"), nframes=1)
    assert areas["nan"] == pytest.approx(exact_area((0, 20), (-30, 30)), rel=0.05)


def test_inputs_dict_list_and_files(tmp_path):
    a, b = strip((-40, -10), (0, 90)), strip((10, 40), (100, 190))
    from_dict = plot_footprints({"A": a, "B": b}, output=None, nframes=1)
    assert list(from_dict) == ["A", "B"]

    from_list = plot_footprints([a, b], output=None, nframes=1)
    assert list(from_list) == ["map 1", "map 2"]

    paths = []
    for name, imap in (("survey_a", a), ("survey_b", b)):
        p = tmp_path / f"{name}_ivar.fits"
        enmap.write_map(str(p), imap)
        paths.append(str(p))
    from_files = plot_footprints(paths, output=None, nframes=1)
    assert list(from_files) == ["survey_a_ivar", "survey_b_ivar"]
    # Same map either way, to within the half-working-pixel wobble that
    # nearest-neighbour resampling gives an edge sitting on a grid line.
    assert from_files["survey_a_ivar"] == pytest.approx(from_dict["A"], rel=0.01)


def test_fullsky_map_covers_the_sphere(tmp_path):
    shape, wcs = enmap.fullsky_geometry(res=2 * utils.degree)
    areas = plot_footprints(
        enmap.ones(shape, wcs), labels=["all sky"], output=str(tmp_path / "a.png"), nframes=1
    )
    assert areas["all sky"] == pytest.approx(41253.0, rel=0.01)


@pytest.mark.parametrize("nx", [10800, 9999])  # divides the working grid, or not
def test_wrapping_map_has_no_seam(nx):
    """A map going all the way round in RA must land on the grid unbroken.

    Its RA=180 edge is not an edge of the sky, so no column of the working grid
    may come out empty there; nx=9999 also leaves the downgrade a part-block to
    crop, which used to widen the same gap.
    """
    from somapy import io as soma_io

    box = np.array([[-60, 180], [20, -180]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=360.0 / nx * utils.degree, proj="car")
    gshape, gwcs = enmap.fullsky_geometry(res=0.2 * utils.degree, proj="car")
    mask = soma_io._footprint(enmap.ones(shape, wcs), 0.0, gshape, gwcs, 0.2)
    rows = mask.sum(0)
    assert rows.min() == rows.max() > 0


def test_partial_map_does_not_wrap_round_the_sky():
    """The seam padding must not carry a map's own edge round to the far side."""
    from somapy import io as soma_io

    gshape, gwcs = enmap.fullsky_geometry(res=0.2 * utils.degree, proj="car")
    mask = soma_io._footprint(strip((-10, 10), (-40, 40)), 0.0, gshape, gwcs, 0.2)
    ra = (
        gwcs.wcs.crval[0] + (np.nonzero(mask.any(0))[0] + 1 - gwcs.wcs.crpix[0]) * gwcs.wcs.cdelt[0]
    )
    assert abs(ra).max() == pytest.approx(40, abs=0.3)


def test_downgrade_preserves_thin_coverage(tmp_path):
    """Max-pooling to the working grid must not erase a sub-pixel-wide strip."""
    thin = strip((-0.05, 0.05), (-90, 90), res=0.5)  # far finer than res=0.5 deg
    areas = plot_footprints(thin, labels=["thin"], output=None, nframes=1, res=0.5)
    assert areas["thin"] > 0


def test_writes_gif_and_still(tmp_path):
    from PIL import Image

    maps = {"A": strip((-40, 0), (0, 120)), "B": strip((0, 40), (120, 240))}

    gif = tmp_path / "spin.gif"
    plot_footprints(maps, output=str(gif), nframes=4, size=180, title="t")
    with Image.open(gif) as im:
        assert im.n_frames == 4
        assert im.size[0] == 180

    png = tmp_path / "still.png"
    plot_footprints(
        maps, output=str(png), nframes=1, size=180, galactic_plane=True, ra_units="deg", ra_sign=1
    )
    assert png.stat().st_size > 0


def test_rotation_actually_changes_the_view(tmp_path):
    """Guard the RA index shift: frames must differ, and a full turn must close."""
    from somapy import io as soma_io

    maps = {"A": strip((-40, 0), (0, 60))}
    frames = []
    real_write = soma_io._write_gif
    soma_io._write_gif = lambda f, *a, **k: frames.extend(f)
    try:
        plot_footprints(maps, output=str(tmp_path / "s.gif"), nframes=8, size=140)
    finally:
        soma_io._write_gif = real_write
    assert len(frames) == 8
    assert not np.array_equal(frames[0], frames[2])
    again = plot_footprints(maps, output=None, nframes=1, size=140, ra0=360.0)
    assert again  # a 360 deg turn is a no-op, not an index error


def test_mismatched_labels_and_colors_are_rejected():
    a = strip((-10, 10), (0, 30))
    with pytest.raises(ValueError, match="labels"):
        plot_footprints([a], labels=["x", "y"], output=None, nframes=1)
    with pytest.raises(ValueError, match="colours"):
        plot_footprints([a, a], colors=["#ffffff"], output=None, nframes=1)


def test_backdrop_is_drawn_under_the_footprints(tmp_path):
    """A sky texture must change the bare sky and leave the fills alone."""
    from somapy import io as soma_io

    sky = enmap.fullsky_geometry(res=2 * utils.degree)
    dec = enmap.posmap(*sky)[0] / utils.degree
    texture = enmap.enmap(np.exp(-0.5 * (dec / 20.0) ** 2) + 0.01, sky[1])
    maps = {"A": strip((-40, 0), (0, 120))}

    real_draw = soma_io._draw
    frames = {}

    def render(name, bd):
        def spy(ax, ids, *a, **k):
            frames[name] = ids.copy()
            return real_draw(ax, ids, *a, **k)

        soma_io._draw = spy
        try:
            plot_footprints(
                maps, output=str(tmp_path / f"{name}.png"), nframes=1, size=200, backdrop=bd
            )
        finally:
            soma_io._draw = real_draw

    render("plain", None)
    render("dust", texture)
    # Same footprint either way: the backdrop paints sky, not coverage.
    assert np.array_equal(frames["plain"], frames["dust"])
    a = np.asarray(Image.open(tmp_path / "plain.png").convert("RGB"), dtype=int)
    b = np.asarray(Image.open(tmp_path / "dust.png").convert("RGB"), dtype=int)
    assert np.abs(a - b).sum() > 0  # ... but the sky does look different


def test_backdrop_auto_falls_back_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMA_SKY_BACKDROP", str(tmp_path / "nope.fits"))
    areas = plot_footprints(
        strip((-20, 20), (0, 60)), labels=["x"], output=None, nframes=1, size=160, backdrop="auto"
    )
    assert areas["x"] > 0


def test_make_backdrop_rotates_galactic_to_equatorial(tmp_path):
    """The prepared artifact must land the Galactic plane where it belongs."""
    import healpy as hp
    from pixell import coordinates

    from somapy.scripts.build_backdrop import make_backdrop

    nside = 64
    b = 90.0 - np.degrees(hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[0])
    hp.write_map(
        str(tmp_path / "gal.fits"), np.exp(-0.5 * (b / 5.0) ** 2) + 0.01, coord="G", overwrite=True
    )

    omap, got_nside, _ = make_backdrop(str(tmp_path / "gal.fits"), res=1.0)
    assert got_nside == nside
    assert omap.shape == (180, 360)

    gl = np.arange(0.0, 360.0, 15.0)
    eq = coordinates.transform("gal", "equ", np.array([gl, gl * 0]) * utils.degree) / utils.degree
    for ra, dec in zip(*eq, strict=True):
        col = int(
            round(enmap.sky2pix(omap.shape, omap.wcs, np.array([[dec], [ra]]) * utils.degree)[1][0])
        )
        peak = int(np.argmax(np.asarray(omap)[:, col % omap.shape[1]]))
        dec_peak = (
            enmap.pix2sky(omap.shape, omap.wcs, np.array([[peak], [col]], float))[0][0]
            / utils.degree
        )
        assert abs(dec_peak - dec) < 3.0, f"ridge off by {dec_peak - dec:.1f} deg"


def test_many_footprints_warn_about_colour_alone():
    maps = {f"f{i}": strip((-10 * i - 10, -10 * i), (0, 40)) for i in range(4)}
    with pytest.warns(RuntimeWarning, match="colours"):
        plot_footprints(maps, output=None, nframes=1, size=180)


def test_cli_renders_and_reports_areas(tmp_path, capsys):
    """The console script must accept files and print the areas it measured."""
    from somapy.scripts.footprints import main

    src = tmp_path / "patch_ivar.fits"
    enmap.write_map(str(src), strip((-30, 10), (-60, 60)))
    out = tmp_path / "cli.png"

    assert (
        main(
            [
                str(src),
                "-o",
                str(out),
                "--nframes",
                "1",
                "--size",
                "160",
                "--backdrop",
                "none",
                "--galactic-plane",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "patch_ivar" in printed and "deg^2" in printed
    assert out.stat().st_size > 0


def test_cli_rejects_bad_arguments(tmp_path):
    from somapy.scripts.footprints import build_parser, main

    with pytest.raises(SystemExit):  # unknown --ra-units choice
        build_parser().parse_args(["x.fits", "--ra-units", "radians"])
    src = tmp_path / "a_ivar.fits"
    enmap.write_map(str(src), strip((-10, 10), (0, 30)))
    with pytest.raises(ValueError, match="labels"):  # two names, one map
        main([str(src), "-o", str(tmp_path / "x.png"), "--nframes", "1", "--labels", "one,two"])
