# Annotation kinds

Fourteen kinds, one contract. This page is what each stores and how to read it;
spec §6–§9 is the normative version.

## The common header

Every annotation carries the same header, whatever its kind:

```python
ann.kind                   # how it is stored
ann.task                   # what it is for: segmentation, detection, classification
ann.grid_id                # the geometry it lives on
ann.timepoints             # which visits it describes
ann.class_ids              # classes it can express
ann.annotated_class_ids    # classes that were examined (§11.3)
ann.closure                # "explicit" | "implicit"
ann.quality_key, ann.prov  # links into quality and provenance
```

### Coverage

`class_ids` is what the annotation can express: every class its data can
encode, whether or not any voxel carries it. `annotated_class_ids` is what was
looked for.

```python
ann.is_annotated("spleen")   # was this class examined at all?
ann.is_fully_covered         # every class it can express was examined
```

A class in `annotated_class_ids` with zero voxels was **examined and absent** —
a usable negative example. A class in neither was not examined, and its absence
carries no information. Training code that treats the second as a negative is
learning from the annotator's schedule.

### Closure

`closure` says whether a class's `is_a` ancestors hold where it does (§5.4).
Under `"explicit"`, the default, nothing is inferred: a class holds where it is
stored. Under `"implicit"` every ancestor of a stored class holds too, and the
readers answer that way: `dense()`, `contains()`, `voxel_counts()` and
`class_bboxes()` give a class the voxels of every class in `class_ids` it is an
ancestor of, in every encoding, and a classification's `state()` is
`positive` for an ancestor of a class asserted above 0 in the same scope unit.
What is stored stays literal: `labelmap()`, a classification's `value()`,
`labels` and `asserted_class_ids`.

```python
from medh5.labels import LabelClass, LabelSet

kidneys = LabelSet("kidneys", classes=[
    LabelClass(1, "kidney", "Kidney"),
    LabelClass(2, "left_kidney", "Left kidney", parents=[1]),
])
with medh5.create("kidneys.medh5", sample_id="k1") as w:
    w.label_set(kidneys)
    w.add_grid("ct", shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))
    w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="ct", modality="CT")
    w.add_classification("finding", {"left_kidney": 1.0},
                         annotated_classes=["kidney", "left_kidney"],
                         closure="implicit")

with medh5.open("kidneys.medh5") as s:
    finding = s.annotations["finding"]
    assert finding.state("kidney") == "positive"   # a left kidney is a kidney
    assert finding.value("kidney") is None         # nothing was asserted for it
```

Under `"explicit"` the same file would answer `negative` for `kidney`: it was
examined, and nothing was asserted for it.

Leave the ancestor examined and unasserted, as above. Asserting `kidney` 0.0
beside `left_kidney` 1.0 contradicts the inference, since `state()` still
answers `positive` while `value()` reads 0.0, and the validator reports it
(`W916`). Voxel annotations should stay `"explicit"`. One declaring
`"implicit"` is reported (`W918`), because every read of an ancestor then
includes its descendants' voxels.

An **ignore** region marks voxels that must not contribute to a loss in either
direction. It is not a class: in band (`labelmap`, `layers`) its voxels hold
the annotation's `ignore_id` (`65535` unless the header says otherwise), and
under the other encodings it is a separate `mask` annotation:

```python
ann.has_ignore_region      # is any voxel ignored, in band or in the sibling mask?
ann.header.ignore_mask     # the id of a separate mask annotation, or None
ann.ignore_mask(roi=roi)   # labelmap and layers only --- see below
```

The ignore id is not a class, so `dense([65535])` is refused with `E404`
rather than returning a plane — an all-zero mask would be indistinguishable
from "examined, and absent", and the ignored voxels would go into the loss.
The same holds for an `ignore_id` the header declares, and for `0`, background:
`dense()` and `contains()` refuse both (`E404`). A stored `ignore_id` outside
`1..65535`, or one that is also a class id, is refused when the annotation is
opened (`E404`).

Where the region lives depends on the encoding, and the writer decides.
`add_segmentation(..., ignore=region)` stores it in band under `labelmap` and
`layers`, where `ignore_mask()` reads it back; `bitmask`, `instances` and
`probmap` cannot represent a reserved id in band, so for those — and under
every encoding where the region overlaps a class — the writer stores it as a
sibling `mask` annotation named `<ann_id>_ignore` and sets `header.ignore_mask`
to it (§7.7). On those three `ignore_mask()` is not defined.

**Read it with `s.ignore_region(ann_id, roi=...)`**, which answers for every
encoding: it combines the in-band region and the sibling mask, and is all
`False` where there is none. `ann.labelmap()` holds it too, under every
encoding: an ignored voxel reads as the ignore id, never as `0`. See
[Partial labels and coverage](../guides/partial-labels.md#some-voxels-i-cannot-label-either-way).

## Voxel annotations

Five encodings, one API. The encoding is a storage decision; the reading code
does not change.

<!-- illustrative -->
```python
ann.contains(class_id, (z, y, x))       # one voxel, one class
ann.dense(["liver", "lesion"])          # (C, *spatial) bool stack
ann.dense(["liver"], roi=(slice(0,64),)*3)
ann.labelmap()                          # (*spatial) class ids, priority order
ann.voxel_counts()                      # {class_id: count}
ann.class_bboxes()                      # {class_id: (S, 2) index-space bounds}
```

A `roi` is a window of the grid, one slice per spatial axis, read whole: a
slice with a step other than 1 raises `ValueError`. `labelmap(dtype=...)`
must hold every class id, and the ignore id where a voxel is ignored; a
narrower dtype raises `ValueError` rather than folding two values into one.

| Encoding | Stores | Good for | Overlap | Instances |
|---|---|---|---|---|
| `labelmap` | one integer volume | many disjoint classes | no | no |
| `layers` | a stack of integer volumes | a few overlapping classes | yes | no |
| `bitmask` | packed bit planes | many classes, heavy overlap | yes | no |
| `instances` | per-object masks with ids | counting, tracking, per-object metrics | yes | yes |
| `probmap` | float per class | soft labels, model output | yes | no |

A sixth voxel kind, `mask`, is a single boolean volume with no classes — the
shape of a field-of-view mask or a stored ignore region, not a segmentation:

<!-- illustrative -->
```python
w.add_mask("fov", fov, grid="ct")                      # bool, grid-shaped
w.add_image("CT", ct, grid="ct", modality="CT", valid_mask="fov")
```

`probmap` answers `contains()` against a **threshold**: 0.5 unless the
annotation declares one. `add_segmentation(..., probabilities=..., threshold=0.3)`
writes it as the `threshold` attribute (spec §7.5); it is covered by
`content_id`, so two readers of one file decide alike.

### Choosing one

You do not have to. `add_segmentation(encoding="auto")` measures the class
overlap graph, colours it greedily, and costs each candidate:

```python
kind, stats = w.add_segmentation("organs", grid="ct", masks=masks)
kind             # "layers"
stats.edges      # the overlapping class pairs it found
```

`medh5 seg stats FILE ANN` shows the same numbers for an existing annotation,
including what each alternative encoding would cost.

### From a label volume

A model's argmax and a converter's label file are already one integer volume.
**`labelmap=` takes it as it is**: each nonzero value is the class of that id,
or the classes `label_values` gives it, and `ignore_value` marks the ignore
region (§7.7). A value may stand for several classes, as a voxel of an nnU-Net
region's component is in its own class and in the region. What it writes is
what the masks the volume implies write, byte for byte, the encoding chosen
from the same `stats`; the masks are never built.

```python
volume = np.select([lesion, liver, spleen], [3, 1, 2]).astype(np.uint8)
kind, stats = w.add_segmentation(
    "segmented",
    grid="ct",
    labelmap=volume,
    label_values={1: "liver", 2: "spleen", 3: ["liver", "lesion"]},
)
kind             # "layers": a lesion voxel is in the liver as well
```

The volume is borrowed from NumPy, never copied, and read with the GIL
released. A nonzero value it holds that `label_values` does not name is
refused (`E404`), as are `ignore=` (the region is a value here) and a
`threshold`. One mask per class held the volume once per class: at 104 classes
of 384³, 11 GB.

### Transcoding

Any encoding converts to any other without loss of what both can express:

```
$ medh5 seg convert case.medh5 organs --to bitmask --dry-run
```

<!-- illustrative -->
```python
w.transcode_annotation("organs", "bitmask")
```

Going to an encoding that cannot represent something is **refused, not silently
dropped**. Four cases:

| From → to | Why it is refused |
|---|---|
| anything dense → `instances` | A dense encoding records which voxels belong to a class, never which object. The conversion would merge every object of a class into one and mint an `instance_id` belonging to none of them (§7.4). |
| `labelmap`/`layers` carrying an in-band ignore region → `bitmask`/`probmap` | Those express ignore as a separate `mask` annotation (§7.7). Dropping it turns "nobody examined these voxels" into "verified absent". Write the ignore region as its own `mask` and reference it with `ignore_mask=` first. |
| anything → `mask` | A `mask` has no classes, so every class would merge into one volume and the coverage contract would be lost. Build a deliberate union with `encode_mask` and write it with `add_mask`. |
| `instances` → dense | Every voxel survives and every `instance_id` does not — and `instance_id` is what tracking joins on across visits (§7.4). Refused unless you pass `drop_identity=True` (`--drop-identity`), which records a `transcode` activity saying identity was dropped. One-way: identity is not recoverable afterwards. |

`medh5.annotations.voxel.transcode_payload`, which converts an encoded payload
without a file, refuses the same conversions, and carries an in-band ignore
region to `labelmap` and `layers`. `check_roundtrip(payload, kind)` transcodes
a payload to `kind` and says whether every class mask and the in-band ignore
region decode from the result as they did from the payload.

One conversion loses something and is not refused, because §7.6 allows it: a
`probmap` is lossless only under its `threshold` (§7.5). Converted to another
encoding, it keeps the voxels at or above the threshold as the class, and its
probabilities are gone — keep the `probmap` where they still matter.

### Instances

Per-object masks, ids and boxes, from an encoding that carries identity:

<!-- illustrative -->
```python
for obj in ann.instances():
    obj.instance_id, obj.class_id
    obj.mask                     # bool array
    obj.box                      # (S, 2) at voxel edges
```

Ask a `layers` annotation and it tells you what to do instead:

```
MEDH5ValidationError: annotation 'organs' of kind 'layers' does not carry
instance identity, and it cannot be recovered from a dense encoding:
transcoding to `instances` would merge every object of a class into one and
mint an id that belongs to none of them (§7.4). Re-derive the objects from
whatever source had them.
```

`instance_id` is **sample-scoped**, which is what makes it a longitudinal join:
object 7 at baseline and object 7 at follow-up are the same lesion. See
[Longitudinal studies](../guides/longitudinal.md#join-the-objects).

## Geometric annotations

### Boxes

Axis-aligned, `float32`, at **voxel edges**:

```python
w.add_boxes("lesions", boxes, class_ids=["lesion"], grid="ct",
            space="index", scores=[0.91], instance_ids=[7])
```

```python
b = s.annotations["lesions"]
b.boxes[0]              # [[1.5, 5.5], [3.5, 8.5], [4.5, 10.5]]
b.as_slices()[0]        # (slice(2, 6), slice(4, 9), slice(5, 11))
b.object_class_ids, b.scores, b.instance_ids   # one entry per box
```

The columns are read once and returned read-only; copy one to edit it.
`instance_ids` is optional (§8.2). Without it, iterating gives each box its
0-based row as `instance_id`, which joins nothing across visits. A score that
is not a number reads as `None`.

`object_class_ids` is each box's class. `class_ids` is the header's coverage
tuple, as on every annotation: every class the annotation can express, which
holds each box's class and may hold classes no box has (§6.2).

`space="index"` or `space="world"`. The half-voxel offset between a box edge
and a slice bound is the convention written down in spec §8.1, and it is the
one thing to get right when writing a detector.

Index coordinates count the voxels of the annotation's **own** grid.
`as_slices(grid=…)`, `to_index(…, grid=…)` and `to_world(…, grid=…)` with
another grid convert through world when the two grids share a `frame_uid`,
and refuse (`E414`) when they do not — a grid without a frame shares nothing
(§3.3). World coordinates are carried into the other grid's units, so a point
at 1 mm lands on a grid of 0.001 m voxels at index 1, and a grid in another
`coord_system` is refused (`E414`): §3.3 rule 4 compares world coordinates in
one convention only. `Grid.world_into(other, points)` is the same step on its
own. A box drawn on a 4 mm grid and read on a 1 mm grid of the same frame
comes back at the 1 mm grid's indices; before 1.4.2 it came back unchanged,
covering a quarter of the anatomy. Boxes carrying `slice_index` are read on
their own grid only, since the planes they name are that grid's.

**2-D boxes on a slice.** A box with a degenerate axis (`lo == hi`) plus
`slice_index` is the common radiology annotation — a lesion drawn on one slice
of a 3-D study (§8.2). Two lesions, on slices 5 and 9:

```python
boxes = np.array([[[5, 5], [3.5, 8.5], [4.5, 10.5]],      # (N, S, 2), with
                  [[9, 9], [10.5, 16.5], [6.5, 12.5]]])   # lo == hi on axis 0
w.add_boxes("lesions", boxes, class_ids=["lesion", "lesion"], grid="ct",
            space="index", slice_index=[5, 9])
```

`slice_index` is a **per-box column**: one plane for each box, shape `(N,)`, and
each plane inside the grid. All three are enforced by the writer, by
`as_slices()` on read, and by `medh5 validate` — the same rule at all three, so
a file written before the check still fails the same way (`E405`). A short
`slice_index` used to drop the boxes past its end and a plane outside the grid
used to be clamped to the edge, both silently.

### Oriented boxes

```python
w.add_obb("nodules", centers, sizes, rotations, class_ids=["nodule"], grid="ct")
```

`rotations` are rotation matrices: column *k* is the direction of the box's
local axis *k*, and each must be a proper rotation (orthonormal, determinant
+1, to 1e-4 — §8.3). `ann.corners()` gives each box's `2**S` corners,
`center + R @ (±size / 2)`; there is no conversion back from corners.

### Keypoints and points

```python
w.add_keypoints("landmarks", keypoints, keypoint_classes, class_ids,
                grid="ct", visibility=vis, skeleton="spine-17")
w.add_points("fiducials_tp0", points, grid="ct",
             correspondence="fiducials_tp1")
```

Keypoints are structured (a skeleton, per-point classes, visibility); points
are a bare set, used for registration landmarks and TRE. `correspondence` names
the paired `points` annotation — the same landmarks, in the same row order, in
the other frame (§10.6).

### Contours

```python
from medh5.annotations.geometric import Polygon

w.add_contours("rtstruct",
               [Polygon(vertices=xyz, class_id=1, plane=(-1, 0), role="outer"),
                Polygon(vertices=hole, class_id=1, plane=(-1, 0), role="hole")],
               grid="ct", space="world")
```

`plane` is `(axis, index)` in the grid's index space — the axis the polygon
is flat along and the slice it sits on, `(0, k)` for a polygon on slice `k` of
an axial stack, `(-1, 0)` for one that is not in-plane. The RTSTRUCT importer
records it, so `ann.by_plane()` groups an imported structure set by slice.
`role` distinguishes an outer boundary from a hole inside it.

**A class's polygons combine by depth** (§8.6). On one plane, a polygon's
depth is how many of the class's other polygons on that plane enclose it;
polygons of one depth are united, and the region is what lies inside an odd
number of those layers — a hole cuts the region around it, an island inside
the hole adds back, and two outlines that cross without nesting are their
union. Write `role="hole"` exactly for a polygon of odd depth. Readers take
the depth, not the stored role: `ann.roles` and `ann.polygons()` report each
in-plane polygon's role from its depth, the validator reports a stored role
that disagrees as `W919` and one other than 0 or 1 as `E404`, and neither
stops a read. A polygon with `plane=(-1, 0)` takes part in no combination
and keeps the role written for it. Avoid storing crossing outlines of one
class on one plane: a reader that combines them by exclusive-or, as many
RTSTRUCT readers do, cancels their overlap.

An empty polygon list is accepted when `annotated_classes` names what was
looked for: a contours annotation can record a verified negative, exactly as
an empty box set can (§9).

Contours stay contours. Rasterising them is a separate, opt-in step whose rule
is recorded in provenance, because "even-odd fill at voxel centres" is a
decision somebody should be able to find later.

### Polylines

Open curves: vessel centrelines, catheter and applicator paths, needle tracks,
and RTSTRUCT's `OPEN_PLANAR` and `OPEN_NONPLANAR` contours
([format 1.2](../spec/medh5-1.2.md) §3). A `Polyline` is ordered vertices,
each joined to the next and the last not joined to the first, so it encloses
nothing:

```python
from medh5.annotations import Polyline

centreline = np.array([[4.0, 2.0, 2.0], [4.0, 6.0, 3.0], [4.0, 9.0, 7.0]])
track = np.array([[1.0, 8.0, 8.0], [6.0, 9.0, 8.5], [9.0, 9.5, 9.0]])
w.add_polylines("vessels",
                [Polyline(centreline, "liver", plane=(0, 4)),
                 Polyline(track, "liver")],
                grid="ct", instance_ids=[11, 12])
```

`plane` is the `(axis, index)` of the grid plane a planar curve lies in, as for
a contour, and `(-1, 0)` (the default) for one that crosses slices. A polyline
has at least two vertices (`E404`), and its `polyline_offsets` delimit the
concatenated `vertices` (`E408`). The reader is a `PolylinesAnnotation`:
`ann.polyline(i)` is one curve's vertices, `ann.polylines()` every curve with
its class and plane, and `ann.offsets` and `ann.planes` the columns.

A polylines annotation is never rasterised and satisfies neither `seg` nor
`det`, whatever its `task` (`other` by default). **It raises the file's
version to 1.2**: a sample is written at the lowest version its content needs,
so a 1.1 reader reads it as its projection, the curves left out as `W913`, and
removing the last polylines annotation writes the file at the version the rest
needs again (1.2 §2.1).

### Meshes

```python
w.add_mesh("surface", vertices, faces, space="world", normals=normals,
           frame_uid="pseudo:frame-a")
```

`space="world"` names a physical frame, so a `frame_uid` is required — from the
argument or from the grid. A mesh whose coordinates are in millimetres without
saying *whose* millimetres is not reproducible, and the writer refuses it
(`E412`).

## Classification

About the sample, one timepoint, one grid, one region, one instance, or one
slice:

```python
w.add_classification("response", {"progressive": 1.0},
                     scope="sample", timepoints=["tp0", "tp1"])
w.add_classification("birads", {"birads_4": 1.0}, scope="instance",
                     scope_ids=[7], schemes=["BI-RADS"])
```

```python
c = s.annotations["response"]
c.labels          # {class_key: score}
c.scope           # "sample" | "timepoint" | "grid" | "roi" | "instance" | "slice"
c.scope_ids
c.assertions()    # every assertion: class_id, value, scope_id, scheme, scheme_value
```

A **change** label — one that describes a difference between two visits — is a
classification whose `timepoints` names both (`c.is_change_label`). See
[Classification and change labels](../guides/classification.md#change-labels-span-an-interval).

## Label sets

```python
from medh5 import LabelSet, LabelClass

labels = LabelSet("abdomen-v3", version="1.0.0", classes=[
    LabelClass(1, "liver", "Liver", category="organ",
               codes=[{"system": "SCT", "code": "10200004"}]),
    LabelClass(3, "lesion", "Lesion", parents=[1]),
])
w.label_set(labels)
```

`parents` makes a DAG: a lesion is inside the liver, so a model trained on
`liver` can be evaluated against `liver ∪ lesion` without a hand-written
mapping. `LabelSet.digest()` is canonical, so two files either agree on the
vocabulary or they demonstrably do not — which is how `medh5 dataset check`
catches a cohort where class 3 means two different things.

A class's `properties` is its own dict: an edit is kept, and counts in the
set's digest. The set `w.label_set(...)` returns is the document's, as is
`w.document.label_set`, so an edit through either reaches the file:

```python
w.document.label_set["liver"].properties["window"] = [-150, 250]
assert w.document.label_set["liver"].properties["window"] == [-150, 250]
```

[Choose and map a label set](../guides/label-sets.md) is the how-to: picking a vocabulary,
fixing a cohort whose files disagree, and comparing two numberings.

Bundled vocabularies:

```
$ medh5 labels registry list
```

```python
from medh5.labels import load
load("brats-subregions")
```

`medh5.labels.available()` names every vocabulary `medh5.labels.load` takes,
the bundled ones and any registered.

Ids `0` (background) and `65535` (ignore) are reserved.

`brats-subregions` uses the BraTS 2023 numbering: 1 necrotic tumour core,
2 peritumoral edema, 3 enhancing tumour. Id 4 is not a BraTS label. It is a
synthetic `tumor` parent of the three. BraTS releases up to 2021 label enhancing
tumour 4, so relabel 4 to 3 before writing their masks with this vocabulary.
Written unchanged, their enhancing tumour lands on the parent.

## Related

- **[Segmentation](../guides/segmentation.md)** — writing and reading masks.
- **[Detection and boxes](../guides/detection.md)** — boxes without an off-by-one.
- **[Classification and change labels](../guides/classification.md)** — choosing a scope.
- **[Partial labels and coverage](../guides/partial-labels.md)** — `annotated_classes` in full.
- **[Specification §6–§9](../spec/medh5-1.0.md#6-annotations--common-model)** — the normative model.
