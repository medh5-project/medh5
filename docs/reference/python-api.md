# Python API

The core of the package is reachable from the top-level `medh5` namespace;
where a name lives in a sub-package, the example imports it from there.

The package is a layer over the format engine, which is written in Rust
([Rust crate](rust.md)); the classes here are its Python face, typed for
`mypy --strict`, and the arrays they hand back are NumPy arrays.

## Opening

```python
import medh5

with medh5.open("case.medh5") as s:        # -> Sample
    ...

with medh5.open_collection("shard.medh5c") as c:   # -> Collection (Mapping[str, Sample])
    for key, sample in c.items():
        ...

from medh5.collection import open_any
with open_any(path, key=None) as opened:   # a Sample or a Collection, whichever it is
    ...
```

`medh5.open` (also exported as `medh5.open_sample`) is lazy: it parses `/meta`
and opens no arrays. Use it as a context manager, or call `.close()`. It is
read-only; every edit goes through `medh5.amend`, which is copy-on-write.

A file of a later minor version than the package implements opens as its
projection (`s.support` is `"projection"`, [1.1](../spec/medh5-1.1.md) §2.2).
A grid, annotation or transform holding a value only that minor defines is
left out of `s.grids`, `s.images`, `s.annotations`, `s.transforms` and
`s.index`, with whatever depends on it. Looking one up raises
`MEDH5ValidationError` coded `W913`, and `s.summary()["left_out"]` names each
one, as `medh5 info` does.

## Sample

### Identity and structure

```python
s.identity            # Identity: sample_id, subject_id, sex, laterality, bodypart
s.cohort              # Cohort: dataset_id, site_id, scanner_id, group_id,
                      #         acquisition_protocol
s.label_set           # LabelSet, or None
s.document            # SampleDocument — the whole /meta document
s.profiles            # frozenset of conformance profiles
s.version             # "1.0"
s.kind                # "sample" or "collection"
s.content_id          # "sha256:..." or None
s.path                # where it was opened from
s.summary()           # JSON-safe description; what `medh5 info --json` prints
```

### Timepoints

```python
s.timepoints                       # Timeline — indexable by position or id
s.timepoints[0].label              # "baseline"
s.timepoints["tp1"].days_from_baseline
s.timepoints.ids                   # ("tp0", "tp1"), in acquisition order
s.timepoints.interval_days("tp0", "tp1")
s.is_longitudinal                  # len(timepoints) > 1

view = s.at("tp1")                 # a timepoint-scoped view of the whole sample
view.images                        # only tp1's images
view.annotations                   # only tp1's annotations
```

Time is stated once, on the grid, and inherited by everything on it. A
single-timepoint sample may leave `timepoint` off its grids (spec §3.7); the
reader resolves those to the one declared timepoint, so `s.at("tp0")`,
`Image.timepoint`, `Annotation.timepoints` and `s.tracks()` answer the same way
whether or not the writer wrote the attribute.

### Geometry

```python
s.grids                    # {grid_id: Grid}
s.reference_grid           # the sample's principal grid

g = s.grids["ct_tp0"]
g.shape, g.spacing, g.origin, g.direction
g.affine                   # (n+1, n+1) index -> world
g.spatial_shape            # spatial axes only, for a 4-D grid
g.coord_system             # "LPS", one of medh5.geometry.COORD_SYSTEMS
g.timepoint                # "tp0"
g.frame_uid                # frame of reference
g.physical_size            # extent per spatial axis, in g.units
g.world_into(s.reference_grid, [[0.0, 0.0, 0.0]])   # in that grid's units
```

### Images

```python
img = s.images["CT_tp0"]
img.shape, img.dtype, img.chunks, img.nbytes
img.modality               # "CT"
img.value_type             # "quantitative"
img.value_units            # "HU"
img.rescale                # (slope, intercept)
img.levels                 # multiscale pyramid levels

img.read()                                # whole array, stored values
img.read(physical=True)                   # rescale applied
img.read((slice(0, 16), slice(0, 64), slice(0, 64)))   # one block
img.dataset                               # the stored dataset (a view, below)
```

`physical=True` applies `slope` and `intercept`. When an image declares
neither, the stored values are already physical and the flag changes nothing.
A physical read is float32 only where a rescale applies; pass `dtype` for one
type across a cohort (`img.read(physical=True, dtype=np.float32)`).

### Annotations

Every annotation, whatever its kind:

```python
ann = s.annotations["organs_tp0"]
ann.kind                   # "layers" | "labelmap" | "bitmask" | "instances" |
                           # "probmap" | "mask" | "boxes" | "obb" |
                           # "keypoints" | "points" | "contours" | "mesh" |
                           # "classification"
ann.task                   # "segmentation" | "detection" | "classification" | ...
ann.grid_id, ann.grid
ann.timepoints             # which visits it describes
ann.class_ids              # classes it declares
ann.annotated_class_ids    # classes that were examined (§11.3)
ann.is_annotated("spleen") # was this class looked for?
ann.is_fully_covered
ann.classes                # LabelClass objects
ann.quality_key, ann.prov
```

Voxel annotations add:

<!-- illustrative -->
```python
ann.contains(class_id, (z, y, x))
ann.dense(["liver", "lesion"])          # (C, *spatial) bool
ann.dense(["liver"], roi=(slice(0,16),)*3)
ann.labelmap()                           # (*spatial) of class ids
ann.voxel_counts()                       # {class_id: count}
ann.class_bboxes()                       # {class_id: (S, 2) or None}
ann.instances()                          # needs instance identity — see below
```

Two reads span objects, so they are on the sample:

```python
s.ignore_region("organs_tp0")            # (*spatial) bool: the §7.7 ignore region,
                                         # in band or in its sibling mask
s.valid_region("CT_tp0", roi=(slice(0, 8),) * 3)   # the image's valid_mask (§4.4),
                                         # or all True where it declares none
```

These are what the loaders put in `item["ignore"]` and `item["valid"]`; see
[PyTorch and MONAI](torch.md#a-batch).

See [Annotation kinds](annotations.md) for the per-kind API.

### Transforms

```python
s.transforms                       # {transform_id: Transform}
t = s.transform_between("tp0", "tp1")   # resolved through the frame graph
s.resolve_frames(frame_a, frame_b) # the same, between two frame uids; memoised per handle
t.kind                             # "identity" | "affine" | "displacement" |
                                   # "bspline" | "composite"
t.from_frame, t.to_frame
t.is_invertible                    # the file claims the mapping is invertible
t.inverse()                        # the stored inverse; a chain's or composite's
                                   # steps inverted in reverse; or None
t.transform_points(points)         # world -> world, in t.units
# A displacement field, w, also has:
# w.jacobian_determinant(roi)      # det(I + du/dx): the whole field's, at the
#                                  # voxels roi selects (== w.jacobian_determinant()[roi])
# w.folding_fraction(roi)          # the fraction of those voxels with det <= 0

from medh5.transforms import jacobian_determinant, target_registration_error
target_registration_error(t, fixed_points, moving_points)   # {"mean", "max", ...}
jacobian_determinant(field, grid)                           # for a displacement field
```

`transform_between(source, target, *, by=None)` accepts a timepoint id, a grid
id or a frame uid at either end, matched in that order unless `by="timepoint"`,
`"grid"` or `"frame"` says which. A timepoint stands for the frames of all its
grids, and has an answer only when they are one frame: a visit holding several
raises `MEDH5ValidationError` naming its grids and frames, so between such
visits ask by grid (`by="grid"`) or resolve between the frames the images you read are on
([why](../guides/longitudinal.md#train-on-the-pairs)). It searches the frame
graph, composing chains and traversing a link backwards only where its inverse
can be *evaluated* — an affine's or identity's analytic inverse, or a stored
`inverse_id` — not merely where `invertible=True` is declared. It returns
`None` when the two are one frame or no path exists — it does not invent one —
and raises `KeyError` for a key that is not a timepoint, a grid
or a frame of reference in the sample, so a mistyped `"TP1"` is not mistaken for
"no registration exists". [Registration between visits](../guides/registration.md)
covers what `None` means and when to store an inverse.

### Tracking

```python
tracking = s.tracks("lesion")
tracking.timepoints                     # ("tp0", "tp1")
tracking.states(instance_id)            # {timepoint: present|resolved|unexamined}
tracking.state_at(instance_id, "tp1")
tracking.is_new(instance_id)
tracking.is_resolved(instance_id)
tracking.is_persistent(instance_id)
tracking.class_conflicts()              # objects whose class changed between visits
tracking.unexamined()                   # {timepoint: instance ids unexamined there}
tracking.coverage                       # {timepoint: class ids whose absence there is a finding}

for instance_id, track in tracking.items():
    track.volumes                       # {timepoint: volume in its grid's units cubed}
    track.relative_change("tp0", "tp1")
    track.at("tp1")                     # the Observation, or None
```

`resolved` means the class was examined at that visit, outside any ignore
region, and the object was not found; `unexamined`, that nobody looked for the
class there or that an ignore region (§7.7) may hide the object. An annotation
whose ignore region holds any voxel counts towards no visit's `coverage`. See
[Longitudinal](../guides/longitudinal.md#three-states-not-two).

### Integrity

```python
s.verify()                    # VerifyResult
s.verify(partial=["images/CT_tp0"])
s.verify().ok
s.verify().unattested         # paths in covered objects no line binds there
s.verify().unverifiable       # digests on data that has none (§13.1): not checked
s.compute_content_id()        # recompute rather than read the stored one
```

`ok` is `False` when a digest mismatches, when the root does, or — in a file
that declares a `content_id` — when a dataset inside a grid, image, annotation
or transform carries no digest at all, or is reached at a path that is not its
own, the one its digest line names (`unattested`).

### Sampling index

```python
s.index                       # {ann_id: SamplingIndex} — every entry in the file
s.fresh_indices               # the ids whose source_digest still matches (§13.3)
```

A stale entry is ignored by the samplers and the statistics, never trusted; see
[Storage](storage.md#the-sampling-index).

### Sampling patches

```python
from medh5.sampling import PatchSampler, iter_patches

sampler = PatchSampler(8, strategy="balanced")
sampler.draw(s, "organs_tp0", rng=0)               # one Patch
sampler.draws(s, "organs_tp0", n=4, rng=0)         # a list, from one generator
sampler.draws(s, None, n=4, rng=0, grid="ct_tp1")  # every window in ct_tp1
for patch in iter_patches(s, sampler, annotation="organs_tp0", n=10_000, rng=0):
    break                                          # drawn one per next()
```

`iter_patches` yields exactly what `draws` returns for the same seed, but
draws each patch when it is asked for, so a long run holds one patch rather
than all of them. It checks the annotation and the grid when it is called,
before the first patch: a misspelt name fails where it was written, not inside
the loop. `rng` is a seed or a `numpy.random.Generator` (`Seed`); a loop
calling `draw` with the same integer seed draws the same patch every time, so
draw a run from one call.

### Stored objects

`s.root` is the sample's HDF5 root, and `img.dataset`, `ann.group`, `t.group`,
`s.index[...].group` and the writer's `w.handle` are objects under it. They are
the package's own views (`medh5.nodes`), not `h5py` objects:

```python
ds = s.root["images/CT"]          # Dataset: shape, dtype, chunks, filters, attrs
ds.shape, ds.dtype, ds.chunks
block = ds[0:4, :, :]             # one read, NumPy out; numpy.asarray(ds) reads all
ds.attrs["modality"]              # attributes, decoded as spec §2.5 fixes
"images" in s.root, s.root["annotations"].keys()
```

A view lives as long as the sample it came from: after `s.close()`, reading
through one — data, shape, attributes or members — raises `MEDH5FileError`, as
does every other read of the file through the sample: `img.read()` and an
image's properties, `ann.dense()`, `t.transform_points()`, `s.identity`. What
was read while the sample was open is held in memory and keeps answering:
`s.grids`, the `s.images` and `s.annotations` mappings, an annotation's header
fields (`kind`, `class_ids`, `annotated_class_ids`, `grid_id`, `timepoints`,
`label_set`, `prov`) and a transform's `kind`, `from_frame` and `to_frame`.
For HDF5 features the views do not cover, open the file with `h5py`
(`pip install "medh5[h5py]"`); it is plain HDF5.

## Writing

<!-- illustrative -->
```python
with medh5.create("out.medh5", sample_id="c1", subject_id="s1",
                  codec="balanced") as w:
    ...
```

Without `sample_id`, the sample id is the file name up to its first dot, and
`subject_id` defaults to it. Where that is not the file's stem, as for
`case.v2.medh5`, `create` warns (`FutureWarning`): medh5 3.0 names that sample
`case.v2`. Pass `sample_id` (and `subject_id`) to keep the name you want, and
to keep `P001.ct.medh5` and `P001.mr.medh5` one subject. A name with nothing
before its first dot, such as `.hidden.medh5`, gives no id and is refused.

`create` and `amend` both return a `SampleWriter`. It builds a temporary file
beside the target and, on a clean exit, fsyncs it, renames it over the target
and fsyncs the directory, so the file appears complete or not at all; an
exception aborts and leaves nothing behind. Without a `with` block, call
`w.commit()` to finish or `w.abort()` to discard. A commit the validator
refuses (below) changes nothing and leaves the writer open, so the defect can
be mended and `commit()` called again; in a `with` block the refusal aborts.
Committing an aborted writer raises. An amendment stamps what it writes, in
the file's own digest algorithm, and never restamps a digest its source
already failed: a file edited behind the library's back is refused (`E701`,
or `E702` when the edit changed what its `content_id` covers) until its bytes
are restored or `medh5 fix --rewrite-digests` restamps them with a recorded
reason.

### Document

<!-- illustrative -->
```python
w.identity(sex="F", bodypart="abdomen")
w.cohort(dataset_id="d", site_id="site-A", group_id="family-7")
w.add_timepoint("tp1", index=1, label="fu1", days_from_baseline=92,
                date="2026-05-04", study_uid="pseudo:...")
w.label_set(label_set)
w.extra("mytool", {"anything": "json-serialisable"})
w.acquisition("CT_tp0", kvp=120, exposure_mas=180)   # imaging physics only
w.deidentification(method="dicom-psi-profile", date_shift_days=-117)
w.split(set_id="cv5", partition="train", fold=1)     # replaces the same set_id
```

### Provenance and quality

```python
tool = w.software("nnU-Net", "2.4.2")
rad  = w.person("RAD-07")
org  = w.organization("Site A")

act = w.activity("predict", agent=tool, tool="nnUNetv2_predict",
                 params={"fold": "all"}, inputs=["images/CT_tp0"])

w.set_quality("organs_tp0", status="reviewed", reviewed_by=[rad.id])
```

Activity types: `import`, `annotate`, `review`, `predict`, `resample`,
`register`, `derive`, `deidentify`, `transcode`, `other`.

### Grids and images

<!-- illustrative -->
```python
w.add_grid("ct_tp0", shape=(192, 256, 256), spacing=(1.5, 0.8, 0.8),
           origin=(-144.0, -102.4, -102.4), direction=np.eye(3),
           coord_system="LPS", timepoint="tp0",
           frame_uid="pseudo:frame-tp0", patch_hint=(96, 96, 96))

w.add_image("CT_tp0", array, grid="ct_tp0", modality="CT",
            value_type="quantitative", value_units="HU",
            rescale_slope=1.0, rescale_intercept=-1024.0, prov=act)

w.add_pyramid("WSI", [level0, level1, level2],
              grid_levels=["l0", "l1", "l2"], modality="SM")
```

`patch_hint` tells the chunk optimiser what shape you will read.
`add_image` also takes `channel_names` for a channel axis, `window_center` /
`window_width` display presets, and `valid_mask` — the id of a `mask`
annotation delimiting the acquired field of view (§4.4):

<!-- illustrative -->
```python
w.add_mask("fov", fov, grid="ct_tp0")        # a bool volume, no classes
w.add_image("CT_tp0", array, grid="ct_tp0", modality="CT", valid_mask="fov")
```

### Annotations

```python
kind, stats = w.add_segmentation(
    "organs_tp0", grid="ct_tp0",
    masks={"liver": liver, "lesion": lesion},   # or probabilities= or instances=
    encoding="auto",                            # or an explicit kind
    # threshold=0.3,                            # with probabilities=: the contains() cut (§7.5)
    annotated_classes=["liver", "spleen", "lesion"],
    ignore=uncertain_mask,                      # voxels nobody examined (§7.7)
    prov=act, quality={"status": "approved"},
)

w.add_boxes("lesions", boxes, class_ids=["lesion"], grid="ct_tp0",
            space="index", scores=[0.91], instance_ids=[7])
w.add_obb("nodules", centers, sizes, rotations, class_ids=["nodule"], grid="ct")
w.add_keypoints("landmarks", keypoints, keypoint_classes, class_ids, grid="ct")
w.add_points("fiducials_tp0", points, grid="ct",
             correspondence="fiducials_tp1")   # the paired point set (§10.6)
w.add_contours("rtstruct", polygons, grid="ct", space="world")
w.add_polylines("vessels", polylines, grid="ct")  # open curves: format 1.2
w.add_mesh("surface", vertices, faces, space="world", frame_uid="pseudo:frame-tp0")
w.add_classification("response", {"progressive": 1.0}, scope="sample",
                     timepoints=["tp0", "tp1"])   # the interval, not one visit
```

`ignore=` is stored wherever the chosen encoding can hold it. `labelmap` and
`layers` carry it in band; under `bitmask`, `instances` or `probmap` — and under
every encoding when the region overlaps a class — the writer stores it as a
sibling `mask` annotation named `<ann_id>_ignore` and sets `ignore_mask` on the
header, so `encoding="auto"` never decides whether the region survives, and it
reads back equal to what you gave. `ignore_mask=` names a mask you wrote
yourself instead; passing both is refused.

Pass exactly one of `masks=`, `probabilities=` and `instances=`, with
`encoding="auto"` or the encoding the argument implies; anything else is
refused rather than half-honoured. `encoding` is `"auto"` or one of the five
encodings (`medh5.annotations.voxel.TRANSCODABLE`): `"mask"` is refused
(`E404`), having no classes to hold them. `instances=[]` with
`annotated_classes=` records "examined, none found" (§7.4).

A mask is boolean. A floating-point one may hold only 0 and 1 — anything else
is refused (`E411`), by `add_mask` too, rather than read as set wherever it is
not zero — so threshold it first, or pass a `bool` array.

Every `commit()` runs the validator's structural and semantic error rules over
the finished file and refuses to write one it would reject, so a file this
writer produces passes `medh5 validate` at the default level. Grid `units` must
be one of `mm`, `um`, `m`, `px` (§3.2); `medh5.geometry.MM_PER_UNIT` gives the
millimetres in one of each length unit (`px` has none).

### Transforms

```python
w.add_transform("tp0_to_tp1", kind="affine",
                from_frame="pseudo:frame-tp0", to_frame="pseudo:frame-tp1",
                matrix=matrix4x4, invertible=True)

w.add_transform("warp", kind="displacement",
                from_frame="pseudo:frame-tp0", to_frame="pseudo:frame-tp1",
                field=field, field_grid="ct_tp0", vector_space="world",
                field_dtype="auto")      # float16 where it is safe (§10.3)
```

`units` default to those of the grids in the two frames, and a `units` they are
not in is refused (`E506`). `w.remove_transform(name)` drops a transform; what
still names it — another's `inverse_id`, a composite's `components` — is
checked at commit, so one removed and added again under its name keeps them.

### Derived data

<!-- illustrative -->
```python
w.build_index()                          # sampling indices for every voxel annotation
w.build_index(["organs_tp0"], max_coords=8192)
w.transcode_annotation("organs_tp0", "bitmask")
w.remove_annotation("old_seg")           # takes its index with it
w.remove_image("PT_tp1")                 # and its acquisition entry and series UID
w.remove_grid("pt_tp1")
w.set_image_attrs("CT_tp0", modality="CT", window_center=[40.0], value_units=None)
w.remap_frame_uids({"old-frame": "new-frame"})   # grids, transforms, world-space
                                                 # annotations — all at once
w.infer_profiles()                       # the profiles the content satisfies
```

`remove_image` and `remove_grid` refuse an id the file does not hold, and
leave the rest to the commit, so removals and additions may come in any order:
it refuses a grid something still names (`E101`, `E503`), and the last image
(`E201`) or grid (`E111`) going. `set_image_attrs` changes an image's
attributes without rewriting its samples; `None` removes one. It checks
values as `add_image` does (`E203`, `E204`, a finite rescale), refuses to
remove `modality` or `value_type` (`E205`), and does not edit `grid`, `digest`
or a pyramid's attributes. `add_mask` returns its group, as every other
`add_*` does.

## Amending

```python
with medh5.amend("case.medh5") as w:
    w.set_quality("organs", status="approved")
```

Copy-on-write: a new file is built from the old and replaced atomically.
Objects this reader does not understand — a `x_` group, an unknown attribute —
are copied through untouched, so amending never silently drops what it cannot
read. A file it cannot preserve is refused before anything is written: a later
minor version (read as a projection, `MEDH5VersionError`), a profile this
package does not implement (`E007`), or one holding HDF5 references — addresses
in the old file, which no copy preserves (`MEDH5FileError`). So is a file
whose attestation no longer holds: a `content_id` that does not recompute from
what the file stores (`E702`, an edit made outside medh5), or a digest or
`digest_algo` this package cannot read (`E703`). An amendment would restamp
either, attesting bytes nobody verified; `medh5 fix --rewrite-digests --reason`
re-attests the file deliberately, and records that it did.

An amendment that finds its source changed after it read it — replaced by
another writer's commit, or edited in place — refuses to commit
(`MEDH5FileError`, "changed after this amendment read it; nothing was
written"), and so does a `repack` or `recompress` in place: renamed over the
file, the other writer's work was lost without an error. Amend it again.

On POSIX, anything holding the file open across an amend keeps reading the old
version; Windows cannot replace a file that is open, so close every reader of
it before the commit, whose rename otherwise fails and leaves the file as it
was. The file's permission bits, read-only included, and its group are
carried onto the replacement; where you are not in the file's group, the
replacement grants its own group only what it grants everyone. Access control
lists and extended attributes are not carried. Through a symbolic link, the file the link names is rewritten and
the link kept; a file with several hard links is replaced at the path given
only, and its other names keep the old version. A link in a directory anyone
can write to, such as `/tmp`, that neither you nor the directory's owner owns
is not followed: the amend is refused (`MEDH5FileError`), as the system would
refuse to open the file through it.

## Resampling

```python
from medh5.transforms import resample

# The follow-up onto the baseline's grid, through the registration between them
report = resample("case.medh5", "CT_tp1", onto="ct")
report.images[0].image_id                # "CT_tp1_on_ct"

# A coarser grid in the image's own frame, written to a new file
report = resample("case.medh5", ["CT"], spacing=(3.0, 1.6, 1.6),
                  grid_id="ct_coarse", interpolation="nearest",
                  out="coarse.medh5")
```

| Signature | |
|---|---|
| `medh5.transforms.resample(path, image_ids, *, onto=None, spacing=None, grid_id=None, suffix=None, interpolation='linear', fill=0, dtype=None, out=None, annotations=True)` | Resample and record; returns a `ResampleReport` |

`onto` names an existing grid; `spacing` (stored order, one value per spatial
axis) with `grid_id` makes a new grid in the source's frame that keeps its
voxel edges. Grids in one comparable frame are mapped by their affines, any
other pair through the sample's transforms (§10.2), and a pair nothing relates
is refused (`MEDH5ValidationError`), as is a target id already taken.
`interpolation` is one of `RESAMPLE_INTERPOLATIONS`; `nearest` keeps the dtype,
`linear` and `cubic` write `float32` unless `dtype` is given. Voxels whose
source point lies outside the source take `fill`, and the output's
`valid_mask` marks the ones holding data. JSON has no NaN, so a non-finite
`fill` is recorded in the activity as Python spells it (`"nan"`, `"inf"`,
`"-inf"`), which `float()` reads back. The `ResampleReport` carries the
`activity_id` of the one `resample` activity, the target `grid_id` and whether
it is `new_grid`, an entry per image (`ResampledImage`: `source`, `image_id`,
`dtype`, `valid_mask`, `filled_voxels`), the transform path per source grid
(`transforms`) and the new `content_id`. `medh5 resample` is the same call.

The annotations on the source grids follow the images, each written as
`<id><suffix>` with `derived_from` naming its source, the activity in `prov`,
and its classes and coverage unchanged. Voxel encodings move nearest in their
own encoding, an `instances` annotation object by object under its own ids,
and a probability map linearly. Points, keypoints and meshes map exactly; a
box is carried only where the mapping keeps the axes, and a contour only where
its plane lands on a grid plane. Anything that cannot be carried exactly
refuses the whole resample rather than being approximated; `annotations=False`
moves the images alone. Target voxels outside the source were never looked at,
so a class-bearing annotation gets an ignore region over them (§7.7): its own ignore mask carried and widened, or a new
`<output>_ignore` mask. Each carried annotation is a `ResampledAnnotation`
(its `source`, the id written, `kind`, `ignore_mask` and `note`) in the
report's `annotations`, and `not_carried` lists the ones left on their grids.

## Clinical history (format 1.1)

```python
from medh5.clinical import HOUR

with medh5.open("cohort/P-03.medh5") as s:
    s.version                          # "1.1"; imaging-only samples stay "1.0"
    s.support                          # "full", or "projection" for a later minor
    c = s.clinical                     # -> Clinical, or None without the profile
    c.clock                            # Clock(id, reference, origin_description, unit)
    c.events                           # (Event, ...) in stored order
    c.event("lab0").available_us       # inclusive (lo, hi) bounds, or None
    c.documents                        # (DocumentInfo, ...): no text read yet
    c.text("report0_text_v1")          # one document's text, read now
    s.document_text("report0_text_v1") # the same, without reading the events
    c.links                            # (Link, ...)
    chosen = c.select(24 * HOUR)       # -> Selection, strict prospective
    chosen.certified, chosen.event_ids
    chosen.admits("image", "CT_tp0")   # may an input read this payload?
    c.records()                        # -> ClinicalRecords: everything, as one bundle
```

`c.select` raises `MEDH5VersionError` for a later minor version's projection
(`s.support == "projection"`): a certified selection claims clinical-training
support, which 1.1 §2.2 forbids for a version this package does not implement.

| | |
|---|---|
| `Clock.relative(clock_id, origin_description)` | The subject clock every time is measured on |
| `Event(event_id, record_id, kind, temporal_type, status, ...)` | One immutable version; times as `(lo, hi)` microseconds (an `int` is an exact instant) |
| `Document(document_id, text, ...)` | Source text, owned by one `document` event |
| `Link.between(source, relation, target, *, asserted_by=None, span=None, target_annotation_id=None)` | A typed relationship between sample-relative objects; `target_annotation_id` names the annotation an `instance` target is in |
| `SelectionPolicy(...)` | Context window, kinds, plans, limits, ties (contract §3.4) |
| `select(events, links, cutoff_us, policy=None)` | Selection over records not read from a file |
| `augment(path, records, *, out=None)` | Add a history to a 1.0 or 1.1 sample, in place or into `out`; returns the report |
| `strip(path, out)` | The imaging projection, as a new 1.0 file |
| `imaging_events_from_timepoints(path)` | Day-precision imaging events from `days_from_baseline` |
| `utc_us(moment)` | An aware `datetime` as microseconds on a `utc` clock; a naive one is refused |
| `day_bounds(day, tz=None)` | Every instant of a calendar day on a `utc` clock, inclusive; with no zone, the fifty hours it may be somewhere |
| `clock_day(t_us)` | The clock day holding an instant, midnight to the last microsecond before the next |

The writer takes the records one at a time — `w.set_clock(...)`,
`w.add_event(...)`, `w.add_document(...)`, `w.add_link(...)` — or as a
bundle, `w.add_records(records)`; each accepts the dataclass, a dict, or
keywords. See [Clinical history beside the images](../guides/clinical.md).

## Tasks and caches

```python
from medh5.task import TaskManifest

task = TaskManifest.load("cohort/progression.task.json")
task.task_fingerprint                  # the definition; not the rows
report = task.preflight()              # -> Preflight: every row, eligible or why not
report.counts, report.findings
row = report.row("P-01@24h")           # -> RowView
row.events, row.slots["ct"].image_id, row.target.status
```

| Module | |
|---|---|
| `medh5.task` | `TaskManifest`, `SourceRef` (`pin`, `check`), `Slot`, `Target`, `preflight`, `Preflight`, `RowView`, `Finding` |
| `medh5.cache` | `CacheWriter`, `FeatureCache`, `validate_cache` (`CacheReport.stale` / `.corrupt`, and the `.manifest_digest` and `.level` it validated), `fitted_on`, `fitted_on_mismatches`, `build_document_cache`, `HashingTextEncoder` (an encoder has `encode`, `dim` and `header`: [the contract](../guides/clinical-training.md#read-documents)) |
| `medh5.torch` | `ClinicalTaskDataset`, `ConceptVocabulary`, `collate_clinical` ([PyTorch](torch.md#clinical-tasks-format-11)) |

| | |
|---|---|
| `TaskManifest.row_fingerprint(row_id)` | One example's identity: the task, the subject, its sources' pinned `content_id`s and the cutoff (contract §3.2) — what a row feature pins |
| `medh5.task.concept_token(kind, code_system, code)` | An event's concept as a vocabulary keys it: the kind, the coding system and the code, a separator inside either escaped so two concepts never share a token |
| `CacheWriter.add(entry_id, values, *, sources, ...)` | One entry, naming every source version it read; at the patient level also its row, fingerprint, cutoff and the versions it encodes |
| `CacheWriter.add_row(row, values, ...)` | A preflight row's feature, its id, fingerprint, cutoff, admitted versions and sources taken from the `RowView` |
| `TaskManifest.repin(base=None, *, sources=None, dry_run=False, confine=None)` | Pin sources to the samples they name now; returns a `RepinReport` (`changed`, a `Repinned` per source, `unchanged`, `reconciled`, `rows`) |
| `medh5.task.Finding` | One thing wrong with a task, its sources or a cache: its code, a key of `medh5.task.CODES` (contract §9) |
| `TaskManifest.notes` | What the manifest should state and does not, each a `medh5.task.Note`; unlike a `Finding`, never a failure |
| `confine=` | On `preflight`, `reconcile` and `repin`, where sources may be opened from (`medh5.task.Confine`): `True` the manifest's base, a directory, or `False` anywhere |

The contract is [Task and cache contract 1](../spec/task-cache-1.md); the
walk-through is [Train on clinical tasks](../guides/clinical-training.md).

## Collections

```python
medh5.pack(["case.medh5", "case2.medh5"], "pair.medh5c")    # keys: names up to the first dot
medh5.unpack("pair.medh5c", "restored/", keys=["case"])       # -> restored/case.medh5
```

Packing moves chunks as stored bytes, so every member keeps its `content_id`;
see [Storage](storage.md#collections).

## Validation

```python
from medh5.validate import validate_file, validate_paths

report = validate_file("case.medh5", level="strict", profiles=["seg"])
report.ok
report.errors            # [Diagnostic]
report.warnings
report.diagnostics[0].code       # "E102"
report.diagnostics[0].location
print(report.format(verbose=True))
```

Levels: `structural` → `semantic` → `integrity` → `strict`. Codes are stable
API; see spec §15.2 and `medh5.CODES`. `validate_paths` takes several files and
returns one report per file, which is what `medh5 validate` prints.

## Exceptions

```
MEDH5Error
├── MEDH5FileError        (also OSError) — carries .errno, .filename
│   ├── MEDH5FileNotFoundError   (also FileNotFoundError)
│   ├── MEDH5PermissionError     (also PermissionError)
│   └── MEDH5IsADirectoryError   (also IsADirectoryError)
├── MEDH5VersionError
├── MEDH5SchemaError
├── MEDH5ValidationError  (also ValueError) — carries .code
└── MEDH5IntegrityError
```

`MEDH5ValidationError.code` is the §15.2 diagnostic code where one applies,
so a caller can branch on the defect rather than on the message text.

A file that is missing, a directory or unreadable raises the subclass that
says so (`medh5.errors.MEDH5FileNotFoundError`,
`medh5.errors.MEDH5IsADirectoryError`, `medh5.errors.MEDH5PermissionError`),
with the system's `errno` and the `filename`, so `except FileNotFoundError`
catches a missing sample; `str()` is the message alone, as before.

`MEDH5Error` is the base of the errors this package defines, not of every
error it raises. HDF5 failing to read a file's bytes — a damaged chunk, say —
raises `MEDH5IOError`, an `OSError` that is not yet a `MEDH5Error` (in 3.0 it
derives from `MEDH5FileError`); another system call that fails raises the
`OSError` subclass of its `errno`; and lookups and arguments raise the built-in
`KeyError`, `IndexError`, `TypeError` and `ValueError`. Where a damaged file
must not stop a loop, catch `OSError` beside `MEDH5Error`.

## Sub-packages

| | |
|---|---|
| `medh5.torch` | [Datasets, samplers and stitching](torch.md) |
| `medh5.monai` | [MetaTensor adapter, `MedH5Reader` and `LoadMedH5d`](torch.md#monai) |
| `medh5.io` | [Converters](converters.md) |
| `medh5.dataset` | [Cohort manifests, splits, statistics](../guides/cohorts.md); `Manifest.save(path, *, paths="keep")` and `Manifest.load(path, *, base=None)` follow [the manifest's path rule](cohort-manifest.md) |
| `medh5.labels` | Vocabularies: `load`, `register`, `describe`, and `check(paths)`, which groups files by vocabulary as [`medh5 labels check --json`](cli.md#medh5-labels) does |
| `medh5.curation` | [Provenance, agreement, tracking, de-identification](curation.md) |
| `medh5.conformance` | [The conformance suite](../spec/conformance.md) |
| `medh5.clinical` | [The clinical profile](../guides/clinical.md) (format 1.1) |
| `medh5.task`, `medh5.cache` | [Tasks and feature caches](../guides/clinical-training.md) |
| `medh5.storage` | [Codecs, chunking, recompression](storage.md) |

## Related

- **[Write and read your first sample](../tutorials/first-sample.md)** — this API end to end.
- **[How-to guides](../guides/index.md)** — the same calls, arranged by task.
- **[Sample document schema](schema.md)** — the fields the writer writes.
- **[Specification](../spec/medh5-1.0.md)** — the normative model behind it, and
  [1.1](../spec/medh5-1.1.md) for the clinical profile.
