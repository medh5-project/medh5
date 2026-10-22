# Resample onto another grid

Move images, and the annotations drawn on them, onto another grid, and keep a
record that you did.

A sample keeps every image on the grid it was acquired on. Nothing in medh5
resamples on its own: the converters refuse two files that disagree, the
loaders refuse a patch that spans two grids, and agreement refuses two
annotations on different grids. `medh5 resample` is the one place it happens,
when you ask, and the file remembers how.

## When to resample, and when not to

Resample for a **consumer**: the grid a model was trained on, a channel stack
that needs one grid, two raters' masks drawn on different series that you want
to compare. Each of those is a decision about one use of the data, with an
interpolation and a loss attached to it.

Do not resample to tidy the archive. A PET at 4 mm and a CT at 0.7 mm are both
right as acquired; folding one onto the other at import throws away what the
next model may need, and a grid can always be derived from the original but
never the reverse. The source objects are never touched: a resample adds new
objects beside them.

## Onto an existing grid

`--onto` names a grid the sample already has. Grids in one frame of reference
are mapped through their affines; grids in two frames through the sample's
transforms (§10.2), so a follow-up lands on the baseline through the
registration between the visits:

```bash
medh5 resample case.medh5 --image CT_tp1 --onto ct_tp0
```

```python
from medh5.transforms import resample

report = resample("case.medh5", "CT_tp1", onto="ct_tp0")
report.images[0].image_id        # "CT_tp1_on_ct_tp0"
report.transforms                # the transform path, per source grid
```

A pair nothing relates is refused, never assumed aligned: two frames with no
transform between them, a grid that names no frame, or two grids in one frame
whose affines cannot be compared (`W917`).

## Onto a new spacing

`--spacing` with `--grid-id` makes a new grid in the image's own frame, one
spacing per spatial axis in stored order (`z,y,x` for a 3-D grid). It covers
the source's extent from voxel edge to voxel edge, so two 1 mm voxels become one
2 mm voxel over the same interval, and the new grid keeps the source's
direction:

```bash
medh5 resample case.medh5 --image CT --spacing 3,1.6,1.6 --grid-id ct_coarse
```

```python
from medh5.transforms import resample

report = resample("case.medh5", ["CT"], spacing=(3.0, 1.6, 1.6),
                  grid_id="ct_coarse", out="coarse.medh5")
report.new_grid                  # True
```

`--out` writes a new file and leaves the source as it was; without it the
sample is amended in place, copy-on-write.

Interpolation is `linear` (the default), `nearest` or `cubic`. `nearest` keeps
the dtype; the others write `float32` unless `--dtype` names one. A voxel whose
source point falls outside the source takes `--fill`, and the output's
`valid_mask` marks the voxels that hold data.

## The annotations come along

Every annotation on a source grid is carried to the target under the same
suffix, naming its source in `derived_from`, with its classes and its coverage
unchanged: what was looked for on the source was looked for on the copy.

- Masks move nearest, in their own encoding. An `instances` annotation moves
  object by object and keeps its ids, so two lesions of one class stay two. An
  object that has no voxel left on the target is refused, never dropped.
- A probability map moves linearly.
- Points and keypoints map exactly through the same mapping as the voxels:
  the inverse of the path the image is sampled through. A displacement field
  has one only when the file stores it and links the pair with `inverse_id`;
  without it the points are refused, since another transform the other way may
  be another mapping. Both halves of a landmark pair (§10.6) carried together
  name each other's copy. A mesh moves only along an affine path, where its
  faces stay flat.
- A box stays a box only where the mapping keeps the axes, and a contour only
  where its plane lands on a grid plane. A classification of slices or of a
  region is carried only onto the source grid's own voxels. Anything else
  cannot be carried exactly, and the whole resample is refused rather than
  approximated.

Target voxels outside the source were never examined, so background there would
be a negative nobody found. They become the ignore region (§7.7) of every
carried mask: its own ignore mask carried and widened, or a new
`<output>_ignore` mask.

```python
from medh5.transforms import resample

report = resample("case.medh5", "CT_tp1", onto="ct_tp0")
[a.source for a in report.annotations]   # ["organs_tp1"]
with medh5.open("case.medh5") as s:
    s.annotations["organs_tp1_on_ct_tp0"].header.derived_from   # ("organs_tp1",)
```

`--no-annotations` (`annotations=False`) moves the images alone. The report
then lists what stayed on its grid as `not_carried`.

## What it holds in memory

The target is written a chunk at a time: an image, its `valid_mask`, a carried
mask or label map, and a new ignore mask each hold one target chunk, never a
target volume. `linear` and `nearest` read only the part of the source image a
chunk maps into; a carried annotation is read whole, in its stored type. A
probability map is the exception on the target side, built whole on the target
grid before it is written; where that is more than memory holds, pass
`--no-annotations`.

`cubic` costs more. Its B-spline prefilter runs over the whole source before the
first target voxel is sampled, because a coefficient depends on every sample
of its line. It holds two `float64` copies of the source while it filters and
one after: sixteen bytes a source voxel at the peak, so about 1 GiB for a
512×512×256 source, whatever its stored type.

## What the file records

One `resample` activity names everything: the inputs (the source objects, the
target grid and the transforms on the path), the outputs, and the parameters
(the target or spacing, the interpolation, the fill, the dtype, whether the
annotations were carried). Every object written names it in `prov`:

```bash
medh5 prov case.medh5
```

```python
from medh5.transforms import resample

report = resample("case.medh5", "CT_tp1", onto="ct_tp0")
with medh5.open("case.medh5") as s:
    (activity,) = s.document.provenance.activities_by_type("resample")
    activity.params["interpolation"]                 # "linear"
    s.images["CT_tp1_on_ct_tp0"].prov == activity.id  # True
```

The command and `medh5.transforms.resample` write the same bytes, so a pipeline
that resamples in Python and one that runs the command produce the same
`content_id`.

## Why the loaders do not resample

A patch is a window in one grid's index space. A loader that resampled behind
the scenes would make a training decision (an interpolation, a target spacing)
for every epoch and every worker, without recording it and without anyone
having chosen it. So the torch datasets refuse a selection that spans two grids,
and the MONAI reader refuses to stack images from two grids as channels; each
refusal names `medh5 resample`. Resample once, into the file, and train on the
derived image: the choice is then visible, repeatable and attached to the data.

**When MONAI's `Spacingd` is enough.** If the resample is part of the model's
preprocessing (a fixed spacing applied on the fly, never stored, and not
something anyone will audit), `Spacingd` in the transform chain is the right
tool. medh5 hands MONAI the correct affine, so `Spacingd` works unmodified.
Reach for `medh5 resample` when the result needs to outlive the run: a
comparison, a published derivative, or a grid other people will train on.

## Related

- **[Registration between visits](registration.md)**: the transforms a resample
  follows between frames.
- **[What the converters refuse, and why](../explanation/refusals.md)**: what
  they will not do for you.
- **[Command line](../reference/cli.md#medh5-resample)** and
  **[Python API](../reference/python-api.md#resampling)**: every option.
