# Survey footprints

`soma.io.plot_footprints` draws the sky footprint of a set of pixell ivar maps on
a rotating globe, with RA/dec labelled.

```python
from soma.io import plot_footprints

areas = plot_footprints(
    {"ACT DR6": "act_ivar.fits", "SO SAT": "so_ivar.fits", "Deep56": deep56_ivar},
    output="footprints.gif", galactic_plane=True,
)
# {'ACT DR6': 18043.2, 'SO SAT': 16511.9, 'Deep56': 834.7}   square degrees
```

Input can be an `enmap`, a path to a FITS map, a sequence of either, or a
`{label: map}` dict. A pixel is inside the footprint where `ivar > threshold`
(default `0`) in any component; NaNs never count. `output` picks the form: `.gif`
for the animation, `.png`/`.pdf`/`.svg` for a still, `None` to render nothing and
just get the areas back.

Each map is thresholded, max-pooled to the working resolution (so a strip thinner
than one working pixel still survives), and resampled onto one shared all-sky CAR
grid. 

An example that needs no data files:

```
python -m soma.scripts.footprint_demo
```

## The sky backdrop

The globe is painted with a Planck thermal-dust map, rotated to equatorial, as a
dim grey texture under the footprints. `src/soma/planck_dust_equ.fits` ships with
the package (360×720 CAR, float32, 1.0 MB).

### Regenerating it

Download an all-sky HEALPix dust map in Galactic coordinates: the Commander
thermal-dust map at Nside 256 is the smallest that is plenty for this (28 MB;
column 0 is `I_ML`, the dust intensity at 545 GHz in µK_RJ):

```
curl -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_dust-commander_0256_R2.00.fits
python -m soma.scripts.build_backdrop COM_CompMap_dust-commander_0256_R2.00.fits \
       -o src/soma/planck_dust_equ.fits
```

The rotation is done in spherical harmonics
(`pixell.reproject.healpix2map(..., rot="gal,equ")`) with `lmax` matched to the
output grid, so the heavy downgrade is a band limit rather than a decimation and
nothing aliases.

The output resolution is `--res`, **in degrees**, default `0.5`. 