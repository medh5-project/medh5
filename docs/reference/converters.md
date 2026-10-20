# Converters

Every importer and exporter lives in `medh5.io` and behind `medh5 convert`.

```bash
pip install "medh5[dicom]"              # reading DICOM, DICOM SEG and RTSTRUCT
pip install "medh5[dicomseg]"           # writing DICOM SEG (highdicom)
```

NIfTI needs neither: the engine reads and writes it, so `medh5.io.nifti`, the
ITK displacement fields and the nnU-Net exporter run with the package alone,
and the standalone `medh5` binary converts NIfTI without Python.

## Conversion reports

Every conversion returns a report, because the interesting part of an import is
not that it succeeded — it is what it had to work out.

```python
from medh5.io.nifti import from_nifti

report = from_nifti({"CT": "ct.nii.gz"}, "case.medh5", masks={"liver": "liver.nii.gz"})
report.ok
report.of_kind("encoding")       # what encoding was chosen and why
for note in report.notes:
    note.severity                # "decision" | "guess" | "warning" | "info"
    note.kind, note.message, note.detail
```

The distinction that matters:

| | |
|---|---|
| **decision** | determined from the data — "the overlap graph has one edge, so `layers`" |
| **guess** | assumed, because the source did not say — "timepoints ordered by file mtime" |
| **warning** | something is wrong or lost |

`--report FILE` writes it as JSON. A guess is not a failure; it is the thing
you go back and check. A report never replaces a file the command reads or
writes: a `--report` naming one — through `..` or a link included — or lying
inside the tree or nnU-Net dataset the command reads or the nnU-Net dataset it
writes is refused before anything is converted, and so is an output naming one
of the command's inputs.

## What each converter does

Every refusal below is explained in
[What the converters refuse, and why](../explanation/refusals.md); this page is
the commands and the contracts.

### NIfTI

```
$ medh5 convert from-nifti case.medh5 --image CT=ct.nii.gz --mask liver=liver.nii.gz
$ medh5 convert import-seg-nifti case.medh5 --mask spleen=spleen.nii.gz --id organs
$ medh5 convert to-nifti case.medh5 CT out.nii.gz
$ medh5 convert to-nifti case.medh5 CT liver.nii.gz --annotation organs --class liver
```

```python
from medh5.io.nifti import from_nifti, to_nifti, import_seg_nifti
```

**Axis order.** NIfTI is `(x, y, z)`; medh5 is `(z, y, x)`. The array is
transposed and the spacing and direction are permuted to match, so the affine
still describes the same physical volume.

**Coordinate system.** NIfTI is RAS+; medh5 defaults to LPS (as DICOM does). The
conversion is a sign flip on the affine, `diag(-1, -1, 1, 1)`, and nothing else
— no resampling, no reinterpretation. `--coord-system RAS` keeps RAS.

**Round trip.** `from_nifti` → `to_nifti` reproduces the affine and the voxels.
A file with a scale (`scl_slope`/`scl_inter`) comes back as its physical values
by default — floats, with no scale; `--stored` (`physical=False`) writes the
stored voxels and the scale, as the file had them.

**A 2-D file is a plane at z = 0.** A 2-D grid (§3.6) is on the first two world
axes and has no z, so a 2-D NIfTI must lie in the x-y world plane at z = 0
(to 10⁻³ mm), which is where `to_nifti` writes one back. A plane off it, or not
parallel to it, is refused rather than moved; saved as `(x, y, 1)` it imports
as a one-slice volume that keeps its z, and the refusal names the option that
does this for you.

**`plane_as_volume=True` imports a plane where it lies** (`--plane-as-volume`;
`read_nifti`, `from_nifti` and `import_seg_nifti` all take it). Every 2-D
image and mask — planes at z = 0 too, so a cohort comes out uniform — is read
as `(x, y, 1)` before the transpose and keeps its full 4×4 affine, giving a
`(1, y, x)` grid: byte for byte the import of the plane re-saved as
`(x, y, 1)`. The third axis is the affine's third column, which must be
non-zero and orthogonal to the first two (within `ORTHONORMAL_TOL`), or the
file is refused: no normal or thickness is invented. A `plane_as_volume`
decision records each plane's z and normal, and the import activity's params
gain `"plane_as_volume": true` when a plane was reshaped. A 3-D or 4-D file is
read the same with or without it.

**An export says what its numbers are in.** Lengths are written in millimetres
— a grid in `m` or `um` is scaled to them, and an uncalibrated `px` grid gets
NIfTI's "unknown" — and a time axis whose frames are evenly spaced becomes
`pixdim[4]` and `toffset`, in the grid's time unit. Uneven frames cannot be one
temporal step: their times go to a BIDS sidecar's `VolumeTiming` beside the
file (`out.json` for `out.nii.gz`), which `from_nifti` reads back, and the
header's step is left at 0. A channel axis is written as one: with a b-value per
channel, four dimensions and a `.bval`; otherwise NIfTI's vector dimension
(`dim[5]`, vector intent), with per-channel echo and inversion times and flip
angles in the sidecar. An export replaces the per-volume fields and `.bval` an
earlier export left beside the file, and keeps the sidecar's other fields. An
annotation is exported on **its own** grid, whichever image is named.

**An annotation is its labelmap, ignored voxels included.** The whole annotation
is written as its `labelmap()`, where a voxel of the §7.7 ignore region is the
annotation's `ignore_id` (`65535` unless the header says otherwise). One class
(`--class`, `class_key=`) is a 0/1 mask, which has no value for "ignored" and
whose `0` says "examined, absent", so it is refused for an annotation with an
ignore region, and for a class outside the annotation's `annotated_class_ids`,
which nobody examined. The output is named `.nii` or `.nii.gz`; any other
name, or a directory, is refused before anything is written.

**Volumes must share one grid** — shape, spacing, origin and direction, with
lengths compared in millimetres, and frame times where both volumes state them.
**A mask added with `import_seg_nifti` must sit on the sample's grid** in the
same sense, not only match its shape: a mask from another grid is refused
rather than laid onto voxels nobody drew on. `assume_aligned=True` takes its
voxels anyway, for a header known to be wrong, and records the guess.

**`scl_slope` and `scl_inter` become the image's rescale**, the way the DICOM
modality LUT does. The voxels keep the dtype the file stores them in;
`read(physical=True)` applies the scale and `read()` returns the stored values
(§4.2). A mask file carrying a scale of its own is thresholded after it is
applied.

`--assume-geometry` imports a file that declares no spatial mapping, recording
the fallback as a guess. Without it, such a file is refused. A file that states
no spatial unit (NIfTI's `unknown`, the header's default, and what an export of
a `px` grid writes) is read as millimetres, as NIfTI readers commonly do, and
the report records that as a guess too.

**Mask names become class keys** as a label set allows them — lowercase
letters, digits and underscores — and the name as given is the class's
display name: `--mask GTV-1=gtv.nii.gz` is the class `gtv_1`, named `GTV-1`.
Two names that make one key are refused. A mask added with `import_seg_nifti`
is found in the sample's label set by the key its name makes.

**`label_set=` fixes the class ids** (`--label-set FILE`). Minted, a class's id
is its rank among the masks one case has, so two cases of a cohort that
segment different organs number them differently. Given a `LabelSet`, or a
JSON file `medh5.labels.load_file` reads, `from_nifti` matches each mask to the
class of the key its name makes (or of its name), mints nothing, and writes
that label set into the sample; a mask it has no class for is refused (E402).

**A class examined and found absent is named, not dropped.** A class
`annotated_classes=` (`--annotated-classes K1,K2`) names that no mask supplies
was searched for and not found (§11.3). Where the import mints its classes it
mints that one too, after the masks' ids, with a `label_set` decision naming
it; with `label_set=`, or in `import_seg_nifti` into a sample that has a label
set, the class must be in it, and the import is refused (E402) otherwise. The
DICOM SEG and RTSTRUCT importers do the same. It was a bare `KeyError`.

**A sample is one subject**, and NIfTI names none: without `--subject-id`
(`subject_id=`) the sample is a subject of its own, its id the sample's, and
the report records that as a guess (`identity`).

**A 4-D series' extra axis is decided, not assumed.** Cine, DCE and 4-D CT put
time there; multi-b-value DWI and multi-echo series put channels there (§3.6),
and NIfTI's header does not distinguish them. A `.bval` sidecar, a BIDS sidecar,
the intent code or a time unit settles it; otherwise the axis is read as time
and the report records a guess. `--fourth-axis time|channel` (`fourth_axis=`)
overrides it, and b-values or echo times go to `acquisition`. A series stored
on NIfTI's fifth dimension, `(x, y, z, 1, n)`, is where NIfTI-1 puts a voxel's
components: with nothing else to say, it is read as channels, on a guess, and
it never takes frame times from the header, whose temporal step is the empty
fourth dimension's.

### DICOM

```
$ medh5 convert from-dicom /studies out/ --group-by subject
$ medh5 convert from-dicom /studies out/ --modality CT --series 1.2.840...
```

<!-- illustrative -->
```python
from medh5.io.dicom import scan_dicom, read_series, from_dicom, select_series

report = from_dicom("/studies", "out/", group_by="subject", modalities=["CT"])
```

**`out` is a directory** — each sample named for its subject inside it —
when the tree holds more than one sample, when it is a directory already, or
when it ends in a separator (`out/`); otherwise it is the file the one sample
is written to.

**Slices are ordered by geometry** — position projected on the slice normal, not
`InstanceNumber`, which is a display order and is regularly wrong.

**Spacing is measured** from the slice origins, not read from
`SliceThickness`, and every slice must lie within a tenth of a slice, and at
least 0.01 mm, of where the regular stack puts it — or the series is refused,
never resampled. The `slice_spacing` decision records the worst slice's
distance (`max_through_plane_error_mm`); [Import from
DICOM](../guides/import-dicom.md#3-expect-refusals-and-read-them) sets out the
policy.

**Modality LUT is stored, not applied.** `RescaleSlope` and `RescaleIntercept`
become the image's `rescale`, so `read(physical=True)` gives the rescaled
values, in the image's `value_units`, and `read()` gives what the scanner
stored; the `value_scale` decision names those units. A CT's `value_units` is HU unless its
`RescaleType` names another quantity — a derived or multi-energy CT may hold
`MGML` or `Z_EFF` — and then it is that; `US` (unspecified) makes the image
`intensity` with no units. A classic CT omits its type only when it is HU,
so an omitted one is `HU`. An enhanced CT must state it, and it must be HU only
for an original, non-localizer, single-energy frame: an omitted type is `HU`
there, and anywhere else unknown, so the image is `intensity` with no units. A
CT's slices or frames whose types still differ are refused; other modalities'
types are not read.

**A per-slice rescale is refused, or applied when asked.** An image stores one
modality LUT (§4.2), and a PET series often states one per slice; taking the
first slice's for all of them would misscale the rest, so such a series is
refused, naming `per_slice_rescale`. With `per_slice_rescale="apply"`
(`--per-slice-rescale apply`), each slice becomes `stored × slope + intercept`,
computed in float64 and stored as float32 with no rescale attributes; a
`per_slice_rescale` decision lists the distinct slopes and intercepts with the
slices (grid indices along the first axis) that state each, and the import
activity's params gain `"per_slice_rescale": "applied"`. SUV is never
computed, and `value_type` and units do not change. A slice that states no
rescale among slices that do is refused either way, and a series with one
rescale is written byte for byte the same whatever the option.

<!-- illustrative -->
```python
report = from_dicom("/pet", "pet.medh5", per_slice_rescale="apply")
```

**Tags are an explicit allow-list.** Imaging physics goes to `acquisition`;
everything else stays out (§11.4). An attribute is recorded only when every
slice states the same value: under tube-current modulation each slice has its
own `XRayTubeCurrent`, and one value would misdescribe the rest, so the image
records none and the report warns (`acquisition`). The importer does not
de-identify, and writes no de-identification record: the report says so, and
the validator reports such a file (`W903`).

**Enhanced multi-frame objects are read as their stacks.** Enhanced and
legacy-converted CT, MR and PET objects (PS3.3 C.7.6.16) are split into stacks
by `StackID` and by the dimensions of `DimensionIndexSequence` that tell
acquisitions apart. A temporal position, a cardiac or respiratory phase, an
echo time, a diffusion weighting, a flip angle or repetition time, a flow
encoding, a spin labelling, a complex component or a contrast phase always
does, even when each of its values holds one plane, and so does any dimension
taken from a functional group that is the same for all of an acquisition's
planes: temporal position, cardiac or respiratory synchronization, the MR
echo, diffusion, timing, modifier, velocity encoding, spin labelling,
metabolite, receive coil and frame type groups, the CT and PET frame type
groups and contrast usage. A dimension pointer may name such a group itself.
Any other dimension, such as a CT's modulated exposure, splits only planes
that repeat: it does when, within one acquisition, it tells apart frames at
the same position and a value of it is shared by frames at different
positions. So a dimension that numbers frames or planes
(`FrameAcquisitionNumber`) splits nothing, and nor does a value that a
plane-varying attribute repeats over distinct planes (a tube current
alternating between two values). Frames at one position that nothing else
tells apart are split by the one dimension that does, if only one does: a
single-plane time series indexed by `FrameReferenceDateTime`, a frame's time,
which in one volume may as well be each slice's own. An acquisition indexed
only by such a dimension, whose planes motion has shifted so that none
repeats, is read with the others as one stack, which is refused unless its
planes still form a regular grid. Each stack becomes a series, with
`Series.frames` naming its `(file, frame index)` pairs and `Series.stack` the
stack where an object holds several, and is imported as a classic series would
be: every frame read as a slice whose position, orientation, pixel spacing,
rescale and acquisition attributes come from its own functional groups, else
the shared ones, else the object's top level (an enhanced MR's
`EffectiveEchoTime` is its `EchoTime`). Frames are decoded one at a time. A
frame that states no position, a stack that is not a regular grid, and a stack
whose `InStackPositionNumber` runs against its positions or skips a number (it
counts a stack's frames from 1, so a gap is a missing plane) are refused with
the reason. Frames with different rescales follow `per_slice_rescale`. A
concatenation's instances are joined by `ConcatenationUID` in
`InConcatenationNumber` order, and one missing a part is refused: the parts
are counted against `InConcatenationTotalNumber`, and where that optional
total is absent their numbers must run from 1 without a gap. Each part that
states `ConcatenationFrameOffsetNumber` must start where the frames of the
parts before it end, or frames between them are missing. An instance with no
`ConcatenationUID` whose markers make it a part (a number or total above 1, or
a frame offset) is refused rather than read as the whole object, and so is an
instance whose `NumberOfFrames` is absent or disagrees with its per-frame
functional groups, which hold one item per frame. One instance found at two
paths is refused rather than read twice, and so is a frame whose position,
orientation or pixel spacing is not numbers. Enhanced ultrasound volumes and
other multi-frame objects are skipped with a warning, and an object that
carries a patient position but is not an image — an RT Dose grid — with a
note.

**One series does not stop the tree.** A series the importer refuses is left
out with a warning (`series`) naming it and the reason, and the rest are
imported; a subject left with no series gets no file (`subject`), and an
import where no series can be read is refused. A pixel decoder's failure is a
refusal naming the file, not a traceback.

**`MONOCHROME1` is stored as written.** Lower stored values are brighter in
such a series; the importer never rewrites values (§4.2), records the
`PhotometricInterpretation` in `acquisition`, and warns in the report so a
model trained on intensity can invert them.

**Images are named by modality and visit** — `CT_tp0` on grid `ct_tp0`,
`PT_tp1` on `pt_tp1`. A study holding several series of one modality numbers
them in `SeriesInstanceUID` order — `MR_1_tp0`, `MR_2_tp0` — and the report
records which series got which name. The timepoint's `series_uids` is keyed by
those image ids.

**`--group-by subject`** (the default) resolves patient identity across studies
and emits one multi-timepoint sample per patient, falling back to one sample per
study — with a warning and a record — when identity cannot be established.

### DICOM SEG

[Import DICOM SEG and RTSTRUCT](../guides/import-seg-rtstruct.md) walks through an import, step by
step.

```
$ medh5 convert from-dicom-seg seg.dcm case.medh5 --id organs
$ medh5 convert from-dicom-seg seg.dcm case.medh5 --annotated-classes liver,tumor
$ medh5 convert to-dicom-seg case.medh5 organs out.dcm \
    --source ct/1.dcm --source ct/2.dcm ...
```

`to-dicom-seg --class K` (`classes=`) exports only the classes named, each one
the annotation examined; `--series-description TEXT` (`series_description=`)
is the SEG's `SeriesDescription`, a DICOM LO of at most 64 characters, and a
longer one is refused.

**Frames are placed by geometry**, from each frame's `PlanePositionSequence` —
not by frame index. Each frame must lie on one of the grid's slices and start at
its first voxel, and its row and column directions and pixel spacing must be the
grid's; a SEG drawn on another reconstruction is refused (`E405`) rather than
force-fitted. A frame's orientation, pixel spacing and segment are read where
the file keeps them, in its own functional groups or the shared ones; frames
turned or spaced against each other are refused, and so is a frame stating no
pixel spacing or naming no segment. Positions are DICOM's (LPS, millimetres),
converted for an RAS grid or one in other length units; a grid in pixels or
another convention is refused (`E414`). **Segments match by label, not by
number**, and segments that share a label — one `Lesion` segment per lesion is
common — are one class: a `BINARY` SEG's are combined as their union and a
`FRACTIONAL` one's as their maximum, recorded as a `segment_union` decision.
Two segments whose labels make one class but code different concepts, or
carry different type modifiers (Left and Right Kidney, both labelled
`Kidney`), are refused. A `BINARY` or `FRACTIONAL` SEG is read; a `LABELMAP`
one, whose frames hold every segment by pixel value, is refused by name.

**Concepts come in with the segments.** A class minted from a SEG takes its
property type as an ontology binding, with the system spelled as §5.2 spells
it where the designator is one the converters know (`SCT` → `SNOMED-CT`,
`RADLEX` → `RadLex`, `FMA`, `UBERON`, `LN` → `LOINC`, `I10` → `ICD-10`,
`DCM`) and as the file states it where it is not. Its property category is
kept as `properties["dicom_category"]`, and becomes `category` where the
concept stands for one: SCT 49755003 is `lesion`, SCT 260787004 `device`, SCT
309825002 `region`; "Anatomical Structure" is an organ's, a vessel's and a
bone's alike, so it sets none. A laterality modifier becomes `laterality`
(the report's `concepts` decision lists both); a modifier that is not a
laterality has no field to go in, and the report's `modifier` decision names
it.

**Frames of reference come first.** A SEG is matched to the grid that shares
its `FrameOfReferenceUID`, and a grid — matched or named with `--grid` — whose
frame is known and is another is refused (`E414`) before the sample is touched:
equal geometry in two frames is no correspondence, and relating them is
registration, which the importer does not apply. Where the SEG or the grid
states no frame there is nothing to compare; the placement rests on geometry
and the report records a guess. A sample `medh5 scrub` pseudonymised holds its
frames' pseudonyms: `--frame-salt` (`frame_salt=`) gives the scrub's salt, and
the SEG's frame is compared as the scrub wrote it. `frames_agree` asks the same
question on its own: equal, not, or `None` when either frame is unknown.

**A SEG may omit its empty frames.** An import places each frame on the grid by
its own position. `read_dicom_seg`, which has no grid and builds the SEG's own
volume, puts the planes the file carries on the regular stack they sit on —
its step the stated `SpacingBetweenSlices`, else the smallest gap or the
least-squares step, and every plane within a tenth of a step, and at least
0.01 mm, of its place, the bound a DICOM series is held to — with the planes
it omits empty; planes no such stack holds are refused, and
`read_dicom_seg_frames` reads them frame by frame. A SEG of one plane has no
step to measure: its `SpacingBetweenSlices` is the step, else its
`SliceThickness`, as a recorded decision, and one stating neither is refused
rather than given 1 mm.

**Export writes an annotation onto its own grid only**: the source images must
be that grid's slices — rows and columns running the same way at the same
spacing, each image starting at its slice's first voxel — or the export is
refused (`E405`); nothing is resampled. They must be in the grid's frame of
reference where it states one (`E414` otherwise); `--frame-salt` (`frame_salt=`)
compares a frame `medh5 scrub` pseudonymised.

**Export writes the classes the annotation examined.** A segment's 0 says
examined and absent, which a class outside `annotated_class_ids` never was, so
such a class is left out, as an `unexamined` decision, and `classes=` naming
one is refused; an annotation that examined none is refused. Every class went
out, and one nobody looked for became an empty segment — a verified negative
to whoever read it back.

<!-- illustrative -->
```python
from medh5.io.dicom_seg import from_dicom_seg, to_dicom_seg

from_dicom_seg("seg.dcm", "case.medh5", ann_id="organs")
to_dicom_seg("case.medh5", "organs", ["ct/1.dcm", "ct/2.dcm"], "out.dcm")
```

Import preserves overlapping segments and `FRACTIONAL` values. Export writes
`BINARY` by default (`segmentation_type="binary"`), from `dense()`, which for
a `probmap` applies the stored threshold — recorded as a `threshold`
decision. `segmentation_type="fractional"`
(`--segmentation-type fractional`) writes a `probmap`'s probabilities as a
`FRACTIONAL` SEG of `fractional_type` `"PROBABILITY"` (the default) or
`"OCCUPANCY"` (`--fractional-type`), each value stored out of 255 and so
within 1/510; any other kind is refused (`E401`). Its `fractional` decision
gives the largest error, how many voxels moved across the stored threshold,
and says that `threshold` and `normalized` are not carried, a SEG having no
field for either.

| `to_dicom_seg` keyword | Values | Default |
|---|---|---|
| `segmentation_type` | `"binary"`, `"fractional"` | `"binary"` |
| `fractional_type` | `"PROBABILITY"`, `"OCCUPANCY"` | `"PROBABILITY"` |

The values are `medh5.io.dicom_seg.SEGMENTATION_TYPES` and
`medh5.io.dicom_seg.FRACTIONAL_TYPES`; `medh5.io.dicom_seg.MAX_FRACTIONAL_VALUE`
is the 255 a fractional value is stored out of.

**Each segment carries its class's concepts.** The property type is the
class's first code whose system has a designator; the property category
`properties["dicom_category"]`, else the concept its `category` stands for
(`organ`, `vessel`, `bone` → SCT 91723000 "Anatomical Structure", `lesion` →
SCT 49755003, `device` → SCT 260787004, `region` → SCT 309825002); its
`laterality` the type modifier (SCT 7771000, 24028007, 51440002). The
algorithm type follows the annotation's `prov`: a `predict` activity by a
software agent of stated version is `AUTOMATIC`, identified by that software
with family DCM 123110 "Artificial Intelligence"; `annotate` and `review` are
`MANUAL`. DICOM requires all three, so where the sample states none the
export writes the generic one — SCT 91723000, `MANUAL` — and records a guess
(`property_type`, `property_category`, `algorithm_type`) naming the classes.
A more specific concept than the sample states is never written, and a
designator is never invented.

A segment cannot mark an ignore region either — its 0 says examined
and absent — so an annotation that ignores any voxel (§7.7) is refused;
`to_nifti` with `annotation=` writes its labelmap, ignore id included. A SEG
holds one segment per class: the export takes a `labelmap`, `layers`,
`bitmask`, `instances` or `probmap` annotation, and refuses any other kind
(`E401`) — a `mask`, which has no classes, among them.

Writing needs `highdicom` (`pip install "medh5[dicomseg]"`).

### RTSTRUCT

The import, same-named ROIs included, is walked through in
[Import DICOM SEG and RTSTRUCT](../guides/import-seg-rtstruct.md).

```
$ medh5 convert from-rtstruct plan.dcm case.medh5 --id contours
$ medh5 convert from-rtstruct plan.dcm case.medh5 --rasterize
$ medh5 convert from-rtstruct plan.dcm case.medh5 --points-id markers
$ medh5 convert from-rtstruct plan.dcm case.medh5 --open-curves polylines
$ medh5 convert to-rtstruct case.medh5 contours out.dcm \
    --points markers --source ct/1.dcm --source ct/2.dcm ...
$ medh5 convert to-rtstruct case.medh5 contours out.dcm \
    --curves contours_curves --source ct/1.dcm --source ct/2.dcm ...
```

<!-- illustrative -->
```python
from medh5.io.rtstruct import from_rtstruct, to_rtstruct

from_rtstruct("plan.dcm", "case.medh5", rasterize=False)
```

**Contours stay contours** (§8.6), in world coordinates. `--rasterize` is opt-in,
and the rule it used goes into the provenance graph; a class whose contours
enclose no voxel centre is a warning, since its empty mask reads as examined
and absent. The mask is annotation `<id>_mask` unless `--mask-id ID`
(`mask_id=`) names another.

**`--label TEXT` (`label=`) is the export's `StructureSetLabel`**, a DICOM SH
of at most 16 characters. A longer one is refused before anything is read; it
was written, and the file was not valid DICOM.

**Areas import as contours, markers as points, curves as polylines on
request.** A medh5 contour is a closed
polygon, so an ROI drawn as `CLOSED_PLANAR` contours imports, and so does one
drawn as `CLOSEDPLANAR_XOR` contours that are nested or apart on every slice,
where their exclusive-or is the holes the import marks. Exclusive-or contours
that cross are refused, naming the ROI and the slice, and so is an ROI mixing
exclusive-or with another type (PS3.3 C.8.8.6.1). An ROI of `POINT` contours —
a marker — imports as rows of a `points` annotation (§8.5), `--points-id ID`
(`points_id=`, `<id>_points` by default): one row per contour, in the grid's
world coordinates and the contours' frame, of the ROI's class and named for
the ROI, with `task` `other`, since a marker is no detection finding; a POINT
contour of more than one point is refused, and a structure set of points
alone writes no contours annotation. An explicit `annotated_classes=` list
is split between the two: a class drawn as points is examined in the points
annotation, every other class it names in the contours one; `all` applies to
the contours only. An ROI of `OPEN_PLANAR` or `OPEN_NONPLANAR` contours — a
brachytherapy applicator, a vessel centreline — is left out by default with a
warning per type (`contour_type`), so no class is minted or marked examined
for it (`--open-curves omit`, `open_curves="omit"`). `--open-curves polylines`
(`open_curves="polylines"`) imports it as rows of a `polylines` annotation
([format 1.2](../spec/medh5-1.2.md) §3), `--curves-id ID` (`curves_id=`,
`<id>_curves` by default): one polyline per contour, in the grid's world
coordinates and the contours' frame, of the ROI's class, with `task` `other`.
An `OPEN_PLANAR` curve records the grid plane it lies in, found along any
axis to the slice tolerance below; one in no plane (a decision,
`curve_plane`) and every `OPEN_NONPLANAR` curve have `axis = -1`. A curve of
fewer than two points is dropped with a warning, and `annotated_classes=` is
split as for points. The option is opt-in because the import amends the
sample and **a polylines annotation writes it at format 1.2**, which a 1.1
reader reads as its projection, leaving the curves out (`W913`); the report
records the raise as a `version` decision. An ROI mixing areas with points or
curves is refused. A contour stating no type is read as `CLOSED_PLANAR`, on a
recorded guess, and one of fewer than three points is dropped with a warning.

**Every contour lies in a slice of the grid**, along its first axis, within a
tenth of a slice and at least 0.01 mm, the bound the DICOM import holds slices
to. A contour between two slices, across several — drawn on another
orientation, or not planar — or outside the grid is refused, not moved onto a
slice.

**The import reads patient coordinates.** `ContourData` is millimetres in LPS,
and the contours are stored in their grid's world coordinates — scaled for a
grid in other length units, flipped in x and y for an RAS one; a grid in pixels
or another convention is refused (`E414`). Each ROI is in the frame of
reference it states, else the one the file lists: ROIs in several frames are
refused (`E414`) before the sample is opened, the message saying which ROI is
in which, and an ROI whose frame the file does not list is a warning. The
contours go onto the grid that shares the ROIs' frame, or the sample's only
grid, and a grid — matched or named with `--grid` — whose frame is known and
is another is refused (`E414`) before the sample is touched. `frame_salt=`
compares a frame `medh5 scrub` pseudonymised, and the contours then carry the
grid's pseudonym. Where the RTSTRUCT or the grid states no frame, the report
records a guess. Into a sample with a label set, an ROI takes the class its
`ROIName` names, by key or by display name.

**ROIs of one class are combined.** ROIs whose names make one key — two
called "GTV", or "GTV" and "gtv" — are one class, minted or found in the
label set, and the report records a `merged_rois` decision naming their ROI
numbers. Their contours combine as §8.6 defines: each polygon's role is its
nesting depth among the class's polygons on its slice, so outlines of two
ROIs that cross are their union. An import where one ROI's contour lies
inside another's on a slice is refused, naming the ROIs and the slice:
combined, the inner one would be a hole in what both drew as an area.
`--roi NUMBER=KEY` (`roi_classes={NUMBER: KEY}`, repeatable) puts an ROI in
the class of that key instead — minted, or one the sample's label set holds
(`E402` otherwise) — which imports same-named ROIs apart:

```
$ medh5 convert from-rtstruct plan.dcm case.medh5 --roi 2=bladder_inner
```

Each imported polygon records the slice it lies on as `contour_plane`
`(0, k)`, in the grid's index space, so `by_plane()` on the result groups the
structure set the way the planner drew it.

**The export writes patient coordinates**, as DICOM defines them: millimetres,
LPS. A grid in metres is scaled and an RAS one flipped in x and y; a grid in
pixels or another convention is refused, and so are contours with no grid to
give their units. The source images must share one frame of reference, and the
contours must be in it — the annotation's `frame_uid` in world space, its
grid's in index space; `--frame-salt` compares a frame `medh5 scrub`
pseudonymised. Contours that state no frame are recorded as a guess.

Each ROI is named for its class's display name — where every class exported
has its own one that an `ROIName` holds, else for its key — and coloured as the
label set colours the class, red where it gives none; text beyond ASCII is
written as UTF-8, under `SpecificCharacterSet` `ISO_IR 192`. Every contour is
`CLOSED_PLANAR`, a hole being a contour nested inside its outer, which readers
that combine an ROI's contours by exclusive-or or even-odd fill read as medh5
does. **Crossing contours mean their union** in medh5, and those readers cancel
the overlap, so two contours of one class that cross on a plane are a warning
(`crossing_contours`) naming the planes: merge them into one outline where
such a reader matters. `--points ANN` (`points=`) also writes a `points`
annotation, one ROI per class of `POINT` contours of one point each, with
`RTROIInterpretedType` `MARKER` (contour ROIs are `ORGAN`); a row with no
class (`E402`) and a class in both annotations are refused. `--curves ANN`
(`curves=`) writes a `polylines` annotation the same way, a polyline per
contour: `OPEN_PLANAR` where it records a grid plane, `OPEN_NONPLANAR` where
it does not, with an empty `RTROIInterpretedType`.

### nnU-Net v2

```
$ medh5 convert from-nnunet /Dataset001_Liver out/ --coord-system LPS --codec archive
$ medh5 convert to-nnunet /out case1.medh5 case2.medh5 --dataset-name Dataset001_Liver
```

<!-- illustrative -->
```python
from medh5.io.nnunetv2 import from_nnunetv2, to_nnunetv2

from_nnunetv2("/Dataset001_Liver", "out/")
to_nnunetv2(["case1.medh5", "case2.medh5"], "/out", dataset_name="Dataset001_Liver")
```

Each case's channels and per-class masks are bundled into one sample, and each
case is a subject of its own: a dataset of cases names no patients, so the
report records that as a guess (`identity`) — two cases of one patient are
two subjects, and a split by file is subject-safe only if no patient has two.
An image is named for its channel; nnU-Net reads `channel_names` as
normalisation schemes, so two channels may share a name (`zscore`), and those
images are named for their channel index as well (`zscore_0000`).

**Test cases are imported on request.** By default the cases are
`imagesTr`'s, and the `imagesTs` cases left out are counted in a warning
(`test_cases`). With `include_test=True` (`--include-test`) they are
imported too, labeled from `labelsTs` where it has their labels, and every
case records its folder as a §12.3 split claim: set `nnunetv2:<dataset>` (the
dataset directory's name), partition `train` or `test`, `assigned_by` "nnU-Net
v2 dataset layout" and the SHA-256 of `dataset.json` as `manifest_sha256`. A
case in both folders is refused.

**nnU-Net's class ids are kept**, so a model trained against the original
dataset still means the same thing. A file's `scl_slope`/`scl_inter` mean what
they mean to nnU-Net: an image keeps them as its rescale, and label ids are read
after them (a scaling that makes non-integer labels is refused). **Region
labels become §5.1 DAG parents**: a region that is the union of two components
is a class whose components name it as a parent, which is exactly what the
hierarchy is for. **nnU-Net's `ignore` label is the annotation's ignore
region** (§7.7), both ways: never a class, and never background. **A label
volume holds only what `labels` declares**: a value it does not is refused, as
nnU-Net refuses it, rather than read as background for every class. **A label
volume is written as it is**, through `add_segmentation(labelmap=...)`, each
value naming its class and every region holding it: no mask per class is
built, so a case of 104 classes costs its label volume, not 104 copies of it.

The parsed `dataset.json` is stashed in `extra["nnunetv2"]`, so `to-nnunet`
reproduces the original dataset definition rather than inventing one: its
channel names and other fields as stashed, its labels unless a case needs them
extended or renumbered (below), and `numTraining` and `file_ending` those of
the files written. Without a stash a channel is named for its image, and a CT
image `CT`: nnU-Net reads the name as the channel's normalisation, and any name
but `CT` — `CT_tp0`, say — z-scores it.

**The export checks the dataset before it writes a file**, and a refusal of any
case refuses the export with nothing written. nnU-Net reads one
physical space per case and one label table for the dataset, so every channel
and the labels must be one lattice in one frame, convention and unit —
nothing is resampled — and the label table is the union of the cases'
classes, one name per id. **Every case must have examined every class
exported** (`annotated_class_ids`): a class one case never looked for would be
written as its background, a negative nobody observed; `classes=` exports a
subset all of them examined. **A dataset directory that already exists may
hold only what the export writes**: a NIfTI file in `imagesTr`, `labelsTr`,
`imagesTs` or `labelsTs` the export would not write — an earlier export's
dropped case or channel, a case since moved between `imagesTr` and
`imagesTs`, the other file ending — refuses it, since nnU-Net trains on every
case file it finds. An identical re-export replaces its own files, and other
files are left alone. **`replace=True` (`--replace`) replaces a dataset
instead**: the directory must be a directory, not a link, holding a
`dataset.json`; the new dataset is written over it in place, file by file,
`dataset.json` last, and the NIfTI files of those four folders that the
export does not write are removed after it. Every other file is left where it
is. The `replace` note lists the files removed and, as `carried_over`, those
left. Every file an export writes goes through the engine's replace (§14.4):
written beside its name and renamed over it once whole, so an export
interrupted part way leaves each file whole — the old one or the new — but
the dataset a mix of the two until it is run again.

**A split places the cases.** With `split="SET_ID"` (`--split SET_ID`) each
case goes where its claim in that set says (§12.3): `test` to `imagesTs`, its
labels to `labelsTs`; `train` or `val` to `imagesTr`, where nnU-Net
cross-validates. A case the set does not place, places twice, or holds out
(`holdout`, `unassigned`) is refused. Test cases are not counted in
`numTraining`, and an unlabeled case claiming `test` needs no `--unlabeled
test`; an unlabeled case claiming `train` or `val` is refused, even with
`--unlabeled test`, which would move it out of the partition it claims.

**Every file is a name under the dataset.** A case's files are named by its
`sample_id`, which must be a sample key (`[A-Za-z0-9_.-]`, not `.` or `..`),
and no two cases may differ only in case — one would overwrite the other on a
case-insensitive file system; `--dataset-name` is held to the same rule, and
`file_ending=` (`--file-ending`) is `.nii.gz` (the default) or `.nii`: every
file is NIfTI. **Every
case is a training case**: one without the annotation is refused, or with
`unlabeled="test"` (`--unlabeled test`) written to `imagesTs` and left out of
`numTraining`. **Label values are consecutive**, as nnU-Net requires: class ids
with a gap — `{1, 3}`, or a subset `--class` exports — are written as
`1..K` in ascending order, and `dataset.json` records each value's class id as
`medh5_class_ids`, which `from-nnunet` reads back, so the ids survive the round
trip. A stashed `dataset.json` that does not name a class some case carries is
extended with it (a region-based one is refused instead): the class was written
as background.

**A case is written only if its labels come back.** Region labels are painted
in `regions_class_order`, as nnU-Net converts them back; without one, each
class is written as its own value. Each exported label is then read back from
the volume outside the ignore region, and a case whose volume does not give one
back — a region its components do not cover, a voxel two classes share — is
refused before any file is written. These refusals carry no §15.2 code: the
samples are valid, and what they fail is nnU-Net's layout.

### ITK transforms

```
$ medh5 convert from-itk-transform reg.tfm case.medh5 --id tp0_to_tp1 --fixed ct_tp0 --moving ct_tp1
$ medh5 convert from-itk-transform out0GenericAffine.mat case.medh5 --id syn \
      --fixed ct_tp0 --moving ct_tp1 --warp out1Warp.nii.gz --vectors lps
$ medh5 convert to-itk-transform case.medh5 back.tfm --id syn --warp-out back_warp.nii.gz --vectors ras
```

<!-- illustrative -->
```python
from medh5.io.itk import from_itk_transform, read_itk_transform, to_itk_transform

from_itk_transform("reg.tfm", "case.medh5", transform_id="tp0_to_tp1",
                   fixed="ct_tp0", moving="ct_tp1")
to_itk_transform("case.medh5", "back.mat", transform_id="tp0_to_tp1")
read_itk_transform("back.mat").matrix     # LPS mm, x_moving = T(x_fixed)
```

`read_itk_transform` returns a `medh5.io.ItkTransform` (its `name`, `matrix`
and, for a composite, `components`; for a B-spline, `bspline`, a
`medh5.io.ItkBSpline` holding ITK's lattice and coefficients), and
`medh5.io.write_itk_transform(path, transform)` writes one, with no sample
involved.

ITK's convention is the format's (§10.2): a registration of a *moving* image
to a *fixed* one maps fixed points to moving points. **The caller names the
two grids**, in ITK's words, because a transform file names no image; the
transform is stored from the fixed grid's frame to the moving grid's. ITK's
points are LPS millimetres, so the matrix is re-expressed in each grid's
coordinate system and units — an RAS grid flips two axes, a grid in metres
scales the translation — and a grid in `px`, which has no millimetres, is
refused. The importer reads text files (`.tfm`, `.txt`), ITK's HDF5 transform
files (`.h5`, `.hdf5`, `.hdf`) and ANTs' MATLAB `.mat`, holding `AffineTransform`, `MatrixOffsetTransformBase`,
`TranslationTransform`, `IdentityTransform`, `Euler3DTransform` (with its
ZYX order where the file sets it), `Euler2DTransform`,
`VersorRigid3DTransform`, `Similarity3DTransform`, `Similarity2DTransform`,
`ScaleTransform`, `BSplineTransform` of order 1 or 3, or a
`CompositeTransform` of them; the centre in the fixed
parameters folds into the translation. An HDF5 file is read as ITK writes it,
`/TransformGroup/<n>/` with `TransformType`, `TransformParameters` and
`TransformFixedParameters`, through the engine's HDF5 (the converters need no
`h5py`), and refused as its text equivalent would be. Refused: a transform
whose input and output dimensions differ, B-splines of order 0 or 2 (1.0
§10.5's evaluation implements 1 and 3), ITKv3's `BSplineDeformableTransform`
(resave it as ITKv4's `BSplineTransform`), and the other non-linear types.

**A composite is listed in reverse.** ITK applies the last transform it lists
first; §10.5 applies the first component first. Each component is stored as an
`affine` between intermediate frames `pseudo:<id>.mid…`, and a `composite`
`<id>` relates the two grids' frames.

**A B-spline is stored as the `bspline` kind** (1.0 §10.5), not sampled:
ITK's lattice becomes a grid `<id>_cp` in the fixed frame, and its
coefficients the control points, both re-expressed in the fixed grid's
coordinate system and units as a matrix is. Inside the lattice's support the
stored transform maps every point as SimpleITK does, to 2e-13 mm in the
measurement the design rationale gives, where a field sampled at the control
points was millimetres off. Beyond the support, ITK leaves a point where it
is, while §10.5 takes the edge control points. The B-spline must be the first step applied
(listed last in an ITK composite), since its lattice is in the fixed frame, so
it is not combined with `--warp`; alone, it needs the moving grid in the fixed
grid's coordinate system and units.

**A displacement field needs `--vectors`.** Some versions of ITK write a
field's vectors in LPS and others in RAS, and the file does not say which:
SimpleITK 2.5 writes a vector image's components as they are (LPS) under
NIfTI's `vector` intent, and reads a `DISPVECT` file's as RAS. An ANTs pair —
the affine and its `Warp.nii.gz`, the warp applied first — is stored as the
warp from the fixed frame to `pseudo:<id>.mid`, the affine from there to the
moving frame, and a `composite` `<id>` between the grids' frames, which earns
`reg` and is the one shortest route between them. The field is stored on the
fixed grid when its geometry is that grid's, otherwise on a new grid
`<id>_field` in the fixed frame. `--inverse-warp` (ANTs' `InverseWarp`) is
stored linked to the warp by `inverse_id`; its field lies in the intermediate
frame, so it is refused where that would give the fixed grid's visit a second
frame.

The export writes each matrix as `AffineTransform_double_3_3` to a `.tfm`,
`.txt`, `.h5` or `.mat` file (several linear steps as an ITK composite in a
text or HDF5 file, or multiplied into one affine in a `.mat`), a `bspline`
as `BSplineTransform` on its control-point grid (to a `.tfm`, `.txt` or `.h5`;
a `.mat` holds only a matrix), and a displacement applied
first to `--warp-out`, a `DISPVECT` NIfTI (intent 1006) in the `--vectors`
convention. Each file is written to a temporary beside it and renamed over
it.

### 0.x files

```
$ medh5 migrate old/*.medh5 -o new/ --write-labels labels.json
$ medh5 migrate old/*.medh5 -o new/ --label-set labels.json --report migration.json
```

```python
from medh5.io.legacy import build_label_set, migrate, migrate_paths
```

The one-way conversion from the 0.x layout (spec Appendix B): mint one label set
for the whole cohort, review it, then convert against it. Box corners shift by
half a voxel, the encoding is chosen by measurement, and grouping by subject is
opt-in because a 0.x file carries no subject key — each reported per file. The
procedure is [Migrate from 0.x](../guides/migrate-0x.md).

### MEDS and FHIR R4

```
$ medh5 clinical augment case.medh5 --from-meds meds/ --meds-subject 1001 --clock-id ehr --clock utc --timezone UTC
$ medh5 clinical augment case.medh5 --from-fhir bundle.json --fhir-subject Patient/p1
$ medh5 clinical augment-cohort --from-meds meds/ --mapping cohort.csv --timezone UTC
```

```python
from medh5.io.fhir import augment_cohort_from_fhir, augment_from_fhir, read_fhir
from medh5.io.meds import augment_cohort_from_meds, augment_from_meds, read_meds
from medh5.io.report import CohortImport, SubjectImport
```

One subject's clinical history, added to a sample's clinical profile (format
1.1): MEDS rows (`pip install "medh5[meds]"`) or FHIR R4 Observations and
DiagnosticReports. No zone, origin, unit, code or availability is invented,
so the report holds no guess; free text and birth instants are imported only
when asked. The mapping tables and refusals are in
[Import from MEDS or FHIR](../guides/clinical-import.md).

`augment_cohort_from_meds` and `augment_cohort_from_fhir` (a FHIR Bulk Data
export) import a whole cohort from one source, read once, into the samples a
mapping names, each subject exactly as the per-sample importer maps it alone.
They return a `CohortImport`: `subjects`, a `SubjectImport` per mapped
subject (its augmentation and report, or the `error` it was refused with),
`imported` and `refused`, and `report`, the notes on the source as a whole.

`medh5.io.meds.COLUMNS` names the MEDS columns the importer reads,
`medh5.io.meds.BOTH_VALUES` the choices for a row holding both values, and
`medh5.io.meds.BIRTH` the birth code left out unless asked;
`medh5.io.fhir.STATUSES` maps a FHIR status to an event's,
`medh5.io.fhir.COMPARATORS` a quantity's comparator to an event's, and
`medh5.io.fhir.UCUM` is the system a quantity's unit code is read from.

### COCO

Not supported, deliberately —
[why](../explanation/refusals.md#coco).

## Related

- **[Import from DICOM](../guides/import-dicom.md)** — a directory of studies to one file per patient.
- **[Import from NIfTI and nnU-Net](../guides/import-nifti.md)** — volumes and masks as separate files.
- **[Export to other formats](../guides/export.md)** — getting data back out.
- **[Migrate from 0.x](../guides/migrate-0x.md)** — the one-way door.
- **[Import from MEDS or FHIR](../guides/clinical-import.md)** — a clinical history beside the images.
- **[What the converters refuse, and why](../explanation/refusals.md)** — every refusal, with the reasoning.
